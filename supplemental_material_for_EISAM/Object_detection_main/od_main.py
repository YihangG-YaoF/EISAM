import argparse
import os
import time
import json
import random
import numpy as np
from tqdm import tqdm
from math import ceil
from math import floor 
from collections import defaultdict
from PIL import Image
from xml.parsers.expat import model

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import (
    LinearLR,          
    CosineAnnealingLR, 
    SequentialLR       
)
import torchvision.transforms.v2 as v2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVIS, LVISEval, LVISResults 

from opts.FSAM import FriendlySAM 
from opts.SAM import SAM
from opts.GSAM.GSAM import GSAM, disable_running_stats, enable_running_stats 
from opts.GSAM.scheduler import CosineScheduler, ProportionScheduler, LinearScheduler
from opts.EISAM.EISAM import EISAM
from opts.EISAM.EISAM_scheduler import ESAMScheduler
from opts.EISAM.EISAM_scheduler import ESAMrhoScheduler


class CustomLVISDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.lvis = LVIS(annFile)  
        self.ids = list(sorted(self.lvis.imgs.keys()))  
        self.transform = transform

    def _load_image(self, id: int):
        img_info = self.lvis.load_imgs([id])[0]
        coco_url = img_info.get('coco_url', '')
        if not coco_url:
            raise KeyError(f"No coco_url for image {id}")

        file_name = coco_url.split('/')[-1]

        possible_paths = [
            os.path.join(self.root.replace('val2017', 'train2017'), file_name), 
            os.path.join(self.root, file_name),                                  
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return Image.open(path).convert("RGB")

        raise FileNotFoundError(f"Image {file_name} not found in train2017 nor val2017")

    def _load_target(self, id: int):
        return self.lvis.load_anns(self.lvis.get_ann_ids(img_ids=[id]))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)  

        if self.transform:
            boxes = []
            labels = []
            for ann in target:
                bbox = ann['bbox']  
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) 
                labels.append(ann['category_id'])

            target_v2 = {
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            }


            image, target_v2 = self.transform(image, target_v2)

            new_target = []
            for box, label in zip(target_v2['boxes'], target_v2['labels']):
                x1, y1, x2, y2 = box.tolist()
                new_target.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'category_id': int(label)
                })
            target = new_target if new_target else []

        return image, target, id


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image_id = self.ids[index]

        if self.transform is not None:
            boxes = []
            labels = []
            for obj in target:
                bbox = obj['bbox']
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                labels.append(obj['category_id'])

            target_v2 = {
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            }

            image, target_v2 = self.transform(image, target_v2)

            new_target = []
            for box, label in zip(target_v2['boxes'], target_v2['labels']):
                x1, y1, x2, y2 = box.tolist()
                new_target.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'category_id': int(label)
                })
            target = new_target if new_target else []

        return image, target, image_id


class ConstantRhoScheduler:
    def __init__(self, rho):
        self.rho = rho

    def step(self):
        return self.rho

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    images = []
    targets = []
    for img, target_list, image_id in batch:
        images.append(img)
        if target_list:
            boxes = []
            labels = []
            for target in target_list:
                bbox = target['bbox']
                x, y, width, height = bbox
                if width > 0 and height > 0:
                    x_min, y_min = x, y
                    x_max, y_max = x + width, y + height
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(target['category_id'])
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
            targets.append({
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            })
        else:
            targets.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
                'image_id': image_id
            })
    return images, targets

def train_and_evaluate_faster_rcnn(
        # training settings
        dataset='lvis',
        dataset_root='/home/3.84T/fydata/coco2017',
        num_epochs=30,
        batch_size=8,
        patience_ratio=0.1,
        eval_interval_ratio=0.1,
        accumulation_steps=1,
        augment=False,
        train_subset_ratio=1.0,
        seed=42,
        device_id=0,
        num_workers=16,
        optimizer_type='GSAM',
        # base optimizer / SGD
        learning_rate=0.005,
        momentum=0.9,
        weight_decay=0.0005,        
        lr_scheduler='cosine',
        lr_min_ratio=1e-8,
        warmup_ratio=0.05,
        # SAM-Family specific
        rho=0.05,
        adaptive=False,
        # EISAM specific
        s=0.01,
        s_scheduler='cosine',
        s_min_ratio=1e-8,
        # GSAM specific
        gsam_alpha=0.5,
        gsam_lr_scheduler=None,
        gsam_rho_scheduler=None,
        rho_min_ratio=None,
        # FSAM specific
        sigma=1.0,
        lmbda=0.9,
):
    set_seed(seed)

    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")

    if augment:
        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),                                     
            v2.ToDtype(torch.float32, scale=True), 
        ])
    else:
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    dataset = dataset.lower()
    if dataset == 'coco':
        ann_file_train = os.path.join(dataset_root, 'annotations/instances_train2017.json')
        ann_file_val   = os.path.join(dataset_root, 'annotations/instances_val2017.json')
        train_dataset = CustomCocoDetection(
            root=os.path.join(dataset_root, 'train2017'),
            annFile=ann_file_train,
            transform=train_transform
        )
        val_dataset = CustomCocoDetection(
            root=os.path.join(dataset_root, 'val2017'),
            annFile=ann_file_val,
            transform=val_transform
        )
    elif dataset == 'lvis':
        ann_file_train = os.path.join(dataset_root, 'annotations/lvis_v1_train.json')
        ann_file_val   = os.path.join(dataset_root, 'annotations/lvis_v1_val.json')
        train_dataset = CustomLVISDetection(
            root=os.path.join(dataset_root, 'train2017'),
            annFile=ann_file_train,
            transform=train_transform
        )
        val_dataset = CustomLVISDetection(
            root=os.path.join(dataset_root, 'val2017'),
            annFile=ann_file_val,
            transform=val_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    

    if train_subset_ratio < 1.0:
        assert 0 < train_subset_ratio <= 1.0, "train_subset_ratio must be between (0, 1]"
        subset_size = int(len(train_dataset) * train_subset_ratio)
        indices = list(range(len(train_dataset)))
        random.seed(seed)  
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        train_dataset = data_utils.Subset(train_dataset, subset_indices)
        print(f"Using a subset of the training data: {subset_size} / {len(train_dataset.dataset)} ({train_subset_ratio * 100:.1f}%)")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )


    num_batches_per_epoch = len(train_dataloader)
    print(f"Number of batches per epoch: {num_batches_per_epoch}")
    num_optimization_steps_per_epoch = ceil(num_batches_per_epoch / accumulation_steps) 
    print(f"Number of optimization steps per epoch: {num_optimization_steps_per_epoch}")
    total_optimization_steps = num_optimization_steps_per_epoch * num_epochs
    print(f"Total optimization steps: {total_optimization_steps}")
    if dataset == 'lvis':
        model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1').to(device)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1204)
    elif dataset == 'coco':
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=91
        ).to(device)

    model = model.to(device)

    base_optimizer_class = torch.optim.SGD
    warmup_steps = int(total_optimization_steps * warmup_ratio)

    base_optimizer = base_optimizer_class(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    if optimizer_type == 'GSAM':
        base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        
        if gsam_lr_scheduler == 'constant':
            gsam_lr_scheduler = None
        elif gsam_lr_scheduler == 'cosine':
            gsam_lr_scheduler = CosineScheduler(
                T_max=total_optimization_steps,
                max_value=learning_rate,
                min_value=learning_rate * lr_min_ratio,
                warmup_steps=warmup_steps,  
                init_value=learning_rate * lr_min_ratio,
                optimizer=base_optimizer
            )
        elif gsam_lr_scheduler == 'linear':
            gsam_lr_scheduler = LinearScheduler(
                T_max=total_optimization_steps,
                max_value=learning_rate,
                min_value=learning_rate * lr_min_ratio,
                warmup_steps=warmup_steps,
                init_value=learning_rate * lr_min_ratio,
                optimizer=base_optimizer
            )

        if gsam_rho_scheduler == 'constant':
            gsam_rho_scheduler = ConstantRhoScheduler(rho=rho)
        elif gsam_rho_scheduler == 'cosine':
            gsam_rho_scheduler = CosineScheduler(
                T_max=total_optimization_steps,
                max_value=rho,
                min_value=rho * rho_min_ratio,
                warmup_steps=warmup_steps,
                init_value=rho * rho_min_ratio,
            )
        elif gsam_rho_scheduler == 'proportion':
            gsam_rho_scheduler = ProportionScheduler(
                pytorch_lr_scheduler=gsam_lr_scheduler,
                max_lr=learning_rate,
                min_lr=learning_rate * lr_min_ratio,
                max_value=rho,
                min_value=rho * rho_min_ratio
            )
        else:
            raise ValueError("Unsupported rho scheduling method")
        optimizer = GSAM(model.parameters(), base_optimizer, model, gsam_alpha=gsam_alpha, rho_scheduler = gsam_rho_scheduler,
                         adaptive=adaptive, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        if optimizer_type == 'FSAM':
            optimizer = FriendlySAM(model.parameters(), base_optimizer_class, rho=rho, sigma=sigma, lmbda=lmbda,
                                    adaptive=adaptive, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'SAM':
            optimizer = SAM(model.parameters(), base_optimizer_class, rho=rho, adaptive=adaptive,
                            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'ASAM':
            optimizer = SAM(model.parameters(), base_optimizer_class, rho=rho, adaptive=True,
                            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'EISAM':
            optimizer = EISAM(model.parameters(), base_optimizer_class, rho=rho, s=s, adaptive=adaptive,
                            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            # To schedule rho, please consult or modify opts/EISAM/EISAM_scheduler.py.
            if s_scheduler == 'cosine':
                s_scheduler = ESAMScheduler(
                    optimizer, 
                    warmup_ratio=warmup_ratio,
                    warmup_start_factor=s_min_ratio * s,
                    mode='cosine', 
                    T_max=total_optimization_steps,
                    s_min=s_min_ratio * s
                )
            else:
                s_scheduler = None  
        elif optimizer_type == 'SGD':
            optimizer = base_optimizer
        else:
            raise ValueError(f"Not supported optimizers: {optimizer_type}")

        if lr_scheduler == 'cosine':
            cosine_scheduler = CosineAnnealingLR(
                optimizer=base_optimizer,
                T_max=total_optimization_steps - warmup_steps,
                eta_min=learning_rate * lr_min_ratio
            )

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer=base_optimizer,
                start_factor=lr_min_ratio * learning_rate,
                total_iters=warmup_steps
            )
            if lr_scheduler == 'cosine':
                lr_scheduler = SequentialLR(
                    optimizer=base_optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                lr_scheduler = warmup_scheduler
        elif lr_scheduler == 'cosine':
            lr_scheduler = cosine_scheduler
        else:
            lr_scheduler = None

    if dataset == 'coco':
        gt_ann = COCO(ann_file_val)
    else:
        gt_ann = LVIS(ann_file_val)


    def _perform_optimizer_step(model, optimizer, optimizer_type, accum_images, accum_targets, device):
        if optimizer_type == 'SGD':
            optimizer.step()

        elif optimizer_type == 'GSAM':
            optimizer.update_rho_t()
            optimizer.perturb_weights()
            disable_running_stats(model)
            optimizer.zero_grad()
            model.train()
            total_loss2 = 0.0
            for imgs, tgts in zip(accum_images, accum_targets):
                loss_dict_batch = model(imgs, tgts)
                batch_loss = sum(loss for loss in loss_dict_batch.values())
                total_loss2 += batch_loss
            total_loss2.backward()
            enable_running_stats(model)
            optimizer.gradient_decompose(optimizer.alpha)
            optimizer.unperturb()
            optimizer.base_optimizer.step()

        elif optimizer_type in ['EISAM', 'FSAM', 'SAM', 'ASAM']:
            def closure():
                disable_running_stats(model)
                optimizer.zero_grad()
                model.train()
                total_loss2 = 0.0
                for imgs, tgts in zip(accum_images, accum_targets):
                    loss_dict_batch = model(imgs, tgts)
                    batch_loss = sum(loss for loss in loss_dict_batch.values())
                    total_loss2 += batch_loss
                total_loss2.backward()
                enable_running_stats(model)
                return None
            optimizer.step(closure=closure)

        else:
            raise ValueError(f"Not supported optimizers: {optimizer_type}")

    
    def evaluate(model, dataloader, device, gt_ann, dataset='coco'):
        model.eval()
        results = []
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = targets[i]['image_id']
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        results.append({
                            'image_id': int(image_id),
                            'category_id': int(label),
                            'bbox': [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                            'score': float(score)
                        })

        if len(results) == 0:
            print("Warning: There are no detections on the validation set; AP=0 is returned.")
            if dataset == 'coco':
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if dataset == 'coco':
            res_file = f'temp_coco_results_{optimizer_type}.json'
            with open(res_file, 'w') as f:
                json.dump(results, f)
            coco_dt = gt_ann.loadRes(res_file)
            coco_eval = COCOeval(gt_ann, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            os.remove(res_file)
            return (coco_eval.stats[0], coco_eval.stats[1], coco_eval.stats[2],
                    coco_eval.stats[3], coco_eval.stats[4], coco_eval.stats[5])

        else: 
            res_file = f'temp_lvis_results{optimizer_type}.json'
            with open(res_file, 'w') as f:
                json.dump(results, f)
            lvis_results = LVISResults(gt_ann, res_file)
            lvis_eval = LVISEval(gt_ann, lvis_results, iou_type='bbox')
            lvis_eval.run()
            lvis_eval.print_results()
            os.remove(res_file)
            return (lvis_eval.results['AP'], lvis_eval.results['AP50'], lvis_eval.results['AP75'],
                    lvis_eval.results['APs'], lvis_eval.results['APm'], lvis_eval.results['APl'],
                    lvis_eval.results['APr'], lvis_eval.results['APc'], lvis_eval.results['APf'])

    if dataset == 'coco':
        init_ap, init_ap50, init_ap75, init_ap_small, init_ap_medium, init_ap_large = \
            evaluate(model, val_dataloader, device, gt_ann, dataset)
        init_metrics = {
            'init_AP': {'value': init_ap},
            'init_AP50': {'value': init_ap50},
            'init_AP75': {'value': init_ap75},
            'init_AP_small': {'value': init_ap_small},
            'init_AP_medium': {'value': init_ap_medium},
            'init_AP_large': {'value': init_ap_large},
        }
        best_ap = init_ap
        best_metrics = {
            'AP': {'value': init_ap, 'step': 0},
            'AP50': {'value': init_ap50, 'step': 0},
            'AP75': {'value': init_ap75, 'step': 0},
            'AP_small': {'value': init_ap_small, 'step': 0},
            'AP_medium': {'value': init_ap_medium, 'step': 0},
            'AP_large': {'value': init_ap_large, 'step': 0},
        }
    else:
        init_ap, init_ap50, init_ap75, init_ap_small, init_ap_medium, init_ap_large, init_ap_rare, init_ap_common, init_ap_frequent = \
            evaluate(model, val_dataloader, device, gt_ann, dataset)
        init_metrics = {
            'init_AP': {'value': init_ap},
            'init_AP50': {'value': init_ap50},
            'init_AP75': {'value': init_ap75},
            'init_AP_small': {'value': init_ap_small},
            'init_AP_medium': {'value': init_ap_medium},
            'init_AP_large': {'value': init_ap_large},
            'init_AP_rare': {'value': init_ap_rare},
            'init_AP_common': {'value': init_ap_common},
            'init_AP_frequent': {'value': init_ap_frequent}
        }
        best_ap = init_ap
        best_metrics = {
            'AP': {'value': init_ap, 'step': 0},
            'AP50': {'value': init_ap50, 'step': 0},
            'AP75': {'value': init_ap75, 'step': 0},
            'AP_small': {'value': init_ap_small, 'step': 0},
            'AP_medium': {'value': init_ap_medium, 'step': 0},
            'AP_large': {'value': init_ap_large, 'step': 0},
            'AP_rare': {'value': init_ap_rare, 'step': 0},
            'AP_common': {'value': init_ap_common, 'step': 0},
            'AP_frequent': {'value': init_ap_frequent, 'step': 0},
        }

    current_opt_step = 0
    no_improve = 0
    num_batches_processed = 0 
    epoch_metrics = defaultdict(list)

    optimizer.zero_grad()

    patience = ceil(1 / eval_interval_ratio * patience_ratio)
    print(f"patients (steps): {patience}")
    eval_interval_steps = ceil(total_optimization_steps * eval_interval_ratio)
    print(f"Evaluation interval (steps): {eval_interval_steps}")
    total_eval_times = floor(1 / eval_interval_ratio)
    start_time = time.time()
    eval_times = 0
    epoch = 0
    accum_images = []
    accum_targets = []
    while current_opt_step < total_optimization_steps and no_improve < patience:
        epoch += 1
        pbar = tqdm(total=len(train_dataloader), desc=f"current_opt_step={current_opt_step}, epoch={epoch}, total_optimization_steps={total_optimization_steps}, eval_times={eval_times}, eval_interval_steps={eval_interval_steps}, eval_left={eval_interval_steps-current_opt_step%eval_interval_steps}, no_improve={no_improve}, patience={patience}")
        for i, (images, targets) in enumerate(train_dataloader):
            if current_opt_step >= total_optimization_steps:
                break

            start_batch_time = time.time()
            num_batches_processed += 1
            images = [img.to(device) for img in images]
            accum_images.append(images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            accum_targets.append(targets)

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            displayed_loss = losses.item()

            losses.backward()

            if (i + 1) % accumulation_steps == 0:
                _perform_optimizer_step(model, optimizer, optimizer_type, accum_images, accum_targets, device)
                optimizer.zero_grad()
                if optimizer_type == 'GSAM':
                    if gsam_lr_scheduler is not None:
                        gsam_lr_scheduler.step()
                elif optimizer_type == 'EISAM':
                    if s_scheduler is not None:
                        s_scheduler.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                else:
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                accum_images = []
                accum_targets = []
                current_opt_step += 1

                if current_opt_step % eval_interval_steps == 0 or current_opt_step == total_optimization_steps:
                    eval_times += 1
                    pbar.set_postfix({
                        'status': "evaluating"
                    })
                    pbar.refresh()
                    if dataset == 'lvis':
                        ap, ap50, ap75, ap_small, ap_medium, ap_large, ap_rare, ap_common, ap_frequent = \
                            evaluate(model, val_dataloader, device, gt_ann, dataset)
                        current_metrics = {
                            'step': current_opt_step, 
                            'AP': ap,
                            'AP50': ap50,
                            'AP75': ap75,
                            'AP_small': ap_small,
                            'AP_medium': ap_medium,
                            'AP_large': ap_large,
                            'AP_rare': ap_rare,
                            'AP_common': ap_common,
                            'AP_frequent': ap_frequent
                        }
                    else:
                        ap, ap50, ap75, ap_small, ap_medium, ap_large = \
                            evaluate(model, val_dataloader, device, gt_ann, dataset)
                        current_metrics = {
                            'step': current_opt_step, 
                            'AP': ap,
                            'AP50': ap50,
                            'AP75': ap75,
                            'AP_small': ap_small,
                            'AP_medium': ap_medium,
                            'AP_large': ap_large,
                        } 

                    for metric_name in best_metrics:
                        if current_metrics[metric_name] > best_metrics[metric_name]['value']:
                            best_metrics[metric_name]['value'] = current_metrics[metric_name]
                            best_metrics[metric_name]['step'] = current_opt_step 

                    if ap > best_ap:
                        best_ap = ap
                        no_improve = 0
                        pbar.set_postfix({
                            'status': 'training'
                        })
                        pbar.refresh()
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            pbar.set_postfix({
                                'status': 'Trigger early stop'
                            })
                            pbar.refresh()
                            break
                    

                pbar.set_description_str(
                    f"best_AP={best_ap} | Epoch={epoch} | opt_step={current_opt_step} | loss={displayed_loss} | "
                    f"eval_left={eval_interval_steps - current_opt_step % eval_interval_steps} | "
                    f"no_improve={no_improve} | "
                    f"eval_times={eval_times} | "
                    f"total_epochs = {num_epochs} | "
                    f"total_steps = {total_optimization_steps} | "
                    f"total_eval_times={total_eval_times} | "
                    f"eval_interval_steps={eval_interval_steps} | "
                    f"patience={patience}"
                )
                pbar.refresh()
            pbar.update(1) 
            batch_time = time.time() - start_batch_time
            epoch_metrics['batch_times'].append(batch_time)

            if no_improve >= patience or current_opt_step >= total_optimization_steps:
                break 
        
        if (i + 1) % accumulation_steps != 0 and current_opt_step < total_optimization_steps:
            _perform_optimizer_step(model, optimizer, optimizer_type, accum_images, accum_targets, device)
            optimizer.zero_grad()
            if optimizer_type == 'GSAM':
                if gsam_lr_scheduler is not None:
                    gsam_lr_scheduler.step()
            elif optimizer_type == 'EISAM':
                if s_scheduler is not None:
                    s_scheduler.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            else:
                if lr_scheduler is not None:
                    lr_scheduler.step()
            accum_images = []
            accum_targets = []
            current_opt_step += 1

        if no_improve >= patience or current_opt_step >= total_optimization_steps:
            break
        # if epoch % 10 == 0:
        #     torch.cuda.empty_cache()

    pbar.close()
    end_time = time.time()
    total_time = end_time - start_time

    avg_batch_time = sum(epoch_metrics['batch_times']) / len(epoch_metrics['batch_times']) if epoch_metrics['batch_times'] else 0

    current_metrics.update({
        'avg_batch_time': avg_batch_time,
    })

    return {
        'init_metrics': init_metrics,
        'best_ap': best_ap,
        'best_metrics': best_metrics,
        'total_time': total_time,
        'stop_step': current_opt_step,
        'stop_eval': eval_times,
        'hyperparams': vars(args),
        'optimizer': optimizer_type,
        'train_metrics': current_metrics,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiple Sharpness-Aware optimizers and SGD are used to train the Faster R-CNN model")
    # training settings
    parser.add_argument('--dataset', type=str, default='lvis', help='Dataset to use (default: coco)')
    parser.add_argument('--dataset_root', type=str, default='data/coco2017', help='Root directory of the dataset')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--patience_ratio', type=float, default=0.15, help='Early stopping patience ratio, relative to total evaluation次数')
    parser.add_argument('--eval_interval_ratio', type=float, default=0.02,
                        help='Every how many optimization steps to evaluate')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps, effective batch_size = batch_size * accumulation_steps')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                        help='Training data subset ratio (0-1), used for debugging, default 1.0 means using all data')
    parser.add_argument('--seed', type=int, default=42, help='Random number seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID, -1 indicates CPU')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads for data loading')
    parser.add_argument('--optimizer', type=str, default='GSAM', 
                        choices=['SGD', 'GSAM', 'FSAM', 'SAM', 'ASAM', 'EISAM'],
                        help='Optimizer type: SGD, GSAM, FSAM, SAM, EISAM')

    # base optimizer / SGD
    parser.add_argument('--learning_rate', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='Currently only cosine or constant scheduling is supported')
    parser.add_argument('--lr_min_ratio', type=float, default=1e-6,
                        help='Minimum learning rate (used for cosine scheduling)')
    parser.add_argument('--warmup_ratio', type=float, default=0,
                        help='Learning rate scheduling warmup ratio')
        
    # SAM-Family specific
    parser.add_argument('--rho', type=float, default=0.1, help='Sharpness parameter rho')
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive mode')

    # EISAM specific
    parser.add_argument('--s', type=float, default=0.01, help='EISAM step size parameter s')   
    parser.add_argument('--s_scheduler', type=str, default='cosine',
                        help='Currently only cosine or constant scheduling is supported.' \
                        'If you need other scheduling options, please consult or modify opts/EISAM/EISAM_scheduler.py.')
    parser.add_argument('--s_min_ratio', type=float, default=1e-8, help='Minimum s value ratio for EISAM (used in cosine scheduling)')
    '''To schedule rho, please consult or modify opts/EISAM/EISAM_scheduler.py.'''

    # GSAM specific
    parser.add_argument('--gsam_alpha', type=float, default=0.5, help='GSAM alpha parameter')
    parser.add_argument('--gsam_lr_scheduler', type=str, default='cosine',
                    choices=['constant', 'cosine', 'linear', 'poly'],
                    help='Learning rate scheduling method')
    parser.add_argument('--gsam_rho_scheduler', type=str, default='constant',
                        choices=['constant', 'cosine', 'proportion'],
                        help='Rho scheduling method (proportion only valid for GSAM)')
    parser.add_argument('--rho_min_ratio', type=float, default=0.005,
                        help='Minimum rho value (used for cosine/proportion scheduling)')    

    # FSAM specific
    parser.add_argument('--sigma', type=float, default=1.0, help='FSAM sigma parameter')
    parser.add_argument('--lmbda', type=float, default=0.9, help='FSAM lambda parameter')

    args = parser.parse_args()

    result = train_and_evaluate_faster_rcnn(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patience_ratio=args.patience_ratio,
        eval_interval_ratio=args.eval_interval_ratio,
        accumulation_steps=args.accumulation_steps,
        augment=args.augment,
        train_subset_ratio=args.train_subset_ratio,
        seed=args.seed,
        device_id=args.device,
        num_workers=args.num_workers,
        optimizer_type=args.optimizer,

        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_min_ratio=args.lr_min_ratio,
        warmup_ratio=args.warmup_ratio,


        rho=args.rho,  
        adaptive=args.adaptive,

        s=args.s,
        s_scheduler=args.s_scheduler,
        s_min_ratio=args.s_min_ratio,
        
        gsam_alpha=args.gsam_alpha,
        gsam_lr_scheduler=args.gsam_lr_scheduler,
        gsam_rho_scheduler=args.gsam_rho_scheduler,
        rho_min_ratio=args.rho_min_ratio,

        sigma=args.sigma,
        lmbda=args.lmbda,        
    )

    opt_type = args.optimizer 
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset}_{opt_type}_{timestamp}.json"
    save_dir = "results"
    save_path = os.path.join(save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)  

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(
            {opt_type: result},
            f,
            ensure_ascii=False,
            indent=4,
            default=str
        )

    print(f"The result has been saved to: {save_path}")