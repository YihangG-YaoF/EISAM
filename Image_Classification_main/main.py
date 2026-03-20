import numpy as np
import random
import torch
import os
from os.path import join
import argparse
from tqdm import tqdm
from colorama import Fore, Style
import wandb
import json
import math
from prefetch_generator import BackgroundGenerator
import shutil
import time
from torchvision import datasets, transforms

from models import ResNet18, ResNet50, ResNet101
from models import WideResNet, PyramidNet
from opt import SAM
from opt import EISAM
from opt import EISAMSscheduler
from opt import EISAMrhoScheduler
from data import SoftCrossEntropyLoss, get_cifar_dataloader
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    """Disables momentum in BatchNorm layers to prevent running stats updates."""
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

# Enable BatchNorm running statistics updates
def enable_running_stats(model):
    """Restores original momentum in BatchNorm layers to enable running stats updates."""
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


# Load checkpoint function
def load_checkpoint(model, optimizer, lr_sched, logs, best_acc, filename='checkpoint.pth.tar'):
    start_epoch = 0
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_sched = checkpoint['lr_sched']
        logs = checkpoint['logs']
        best_acc = checkpoint['best_acc']
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{filename}'")
    return model, optimizer, start_epoch, lr_sched, logs, best_acc

# Initialize random state
def init_random_state(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Main function
def main():
    parser = argparse.ArgumentParser(description='Training ResNet18 on CIFAR-100 with EISAM')

    # Data configuration
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--data_name', default='CIFAR-100', choices=['CIFAR-10', 'CIFAR-100'])
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--use_cutout', default=False, type=bool)
    parser.add_argument('--length', default=8, type=int)
    parser.add_argument('--use_auto_augment', default=False, type=bool)
    parser.add_argument('--use_rand_augment', default=False, type=bool)
    parser.add_argument('--use_random_erasing', default=False, type=bool)
    parser.add_argument('--use_cutmix', default=False, type=bool)
    parser.add_argument('--cutmix_alpha', default=1.0, type=float)
    parser.add_argument('--use_mixup', default=False, type=bool)
    parser.add_argument('--mixup_alpha', default=1.0, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--swa_start', default=150, type=int)

    # Optimizer configuration
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--use_sam', default=False, type=bool)
    parser.add_argument('--use_eisam', default=False, type=bool)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5.0e-4, type=float)
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'multistep', 'const'])
    parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--decay_gamma', default=0.1, type=float)
    parser.add_argument('--sam_rho', default=0.05, type=float)

    # EISAM hyperparameters
    parser.add_argument('--eisam_rho', default=0.05, type=float)
    parser.add_argument('--eisam_s', default=0.01, type=float)
    parser.add_argument('--adaptive', default=False, type=bool)

    # Training configuration
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--save_dir', default='test', type=str)
    parser.add_argument('--project', default=None)
    parser.add_argument('--restart', default=False, type=bool)
    parser.add_argument('--wandb', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50', 'resnet101', 'wideresnet', 'pyramidnet'])

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if args.project is None:
        args.project = f'{args.arch}_{args.data_name}_'
        if args.use_cutout:
            args.project += 'cutout'
        elif args.use_auto_augment:
            args.project += 'auto_augment'
        elif args.use_rand_augment:
            args.project += 'rand_augment'
        elif args.use_cutmix:
            args.project += 'cutmix'
        elif args.use_mixup:
            args.project += 'mixup'
        else:
            args.project += 'basic'
    args.project
    
    if args.use_sam:
        args.instance = f'sam_rho_{args.sam_rho}_'
    elif args.use_eisam:
        args.instance = f'eisam_rho_{args.eisam_rho}_eisam_s_{args.eisam_s}_'
    else:
        args.instance = 'vanilla_'
    args.instance += f'lr={args.learning_rate}_bs={args.batch_size_train}_wd={args.weight_decay}_seed={args.seed}'

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_dir = join(args.save_dir, args.project, args.instance, timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(join(save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.wandb:
        os.environ["WANDB_API_KEY"] = "xxxxx"
        wandb_run = wandb.init(config=args, project=args.project, name=args.instance)

    init_random_state(args.seed)

    train_loader, test_loader = get_cifar_dataloader(args)

    if args.arch == 'resnet18':
        net = ResNet18(num_classes=args.num_classes).to(device)
    elif args.arch == 'resnet50':
        net = ResNet50(num_classes=args.num_classes).to(device)
    elif args.arch == 'resnet101':
        net = ResNet101(num_classes=args.num_classes).to(device)
    elif args.arch == 'wideresnet':
        net = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10).to(device)
    elif args.arch == 'pyramidnet':
        net = PyramidNet(args.data_name, 110, 270, args.num_classes, False).to(device)
    else:
        raise NotImplementedError(f'Architecture {args.arch} is not implemented!')

    optim_param = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    if args.optimizer == 'SGD':
        base_optimizer = torch.optim.SGD
        optim_param.update({'momentum': args.momentum, 'nesterov': True})
    elif args.optimizer == 'Adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'AdamW':
        base_optimizer = torch.optim.AdamW
    else:
        raise NotImplementedError(f'{args.optimizer} is currently not implemented!')
    
    if args.use_sam:
        optim_param.update({'rho': args.sam_rho})
        optimizer = SAM(net.parameters(), base_optimizer, **optim_param)
    elif args.use_eisam:
        optimizer = EISAM(net.parameters(), base_optimizer, rho=args.eisam_rho, s=args.eisam_s, 
                     adaptive=args.adaptive, **optim_param)
    else:
        optimizer = base_optimizer(net.parameters(), **optim_param)

    rho_scheduler = None
    s_scheduler   = None
    if args.use_eisam:
        # 阶梯衰减（Step Decay），每 50 个 epoch 衰减一次（乘以 gamma）
        # rho_scheduler = EISAMrhoScheduler(
        #     optimizer, 
        #     mode='step', 
        #     step_size=50,     
        #     gamma=0.1
        # )

        # 余弦退火（Cosine Annealing），总共 200 个 epoch 一个周期
        # rho_scheduler = EISAMrhoScheduler(
        #     optimizer, 
        #     mode='cosine', 
        #     T_max=200,        
        #     rho_min=1e-6
        # )
        
        # 带重启的 cosine（初始周期 100，重启后周期翻倍）
        # rho_scheduler = EISAMrhoScheduler(
        #     optimizer, 
        #     mode='cosine_restart', 
        #     restart_period=50,   # 初始周期 T_0
        #     mult_factor=1.0,      # 每个重启周期长度 *2
        #     rho_min=1e-4,
        #     verbose=True
        # )
        
        rho_scheduler = EISAMrhoScheduler(
            optimizer, 
            mode='none'
        )

    
        # 阶梯衰减（Step Decay）
        # s_scheduler = EISAMsScheduler(
        #     optimizer, 
        #     mode='step', 
        #     step_size=50,     
        #     gamma=1.1
        # )

    # 余弦退火（Cosine Annealing），总共 200 个 epoch 一个周期
        s_scheduler = EISAMSscheduler(
            optimizer, 
            mode='cosine', 
            T_max=200,         # 总 epoch 数
            s_min=1e-8
        )
    
        # 带重启的 cosine（初始周期 100，重启后周期翻倍）
        # s_scheduler = EISAMsScheduler(
        #     optimizer, 
        #     mode='cosine_restart', 
        #     restart_period=100,   
        #     mult_factor=2.0,      
        #     s_min=1e-6,
        #     verbose=True
        # )
        
        # s_scheduler = EISAMsScheduler(
        #     optimizer, 
        #     mode='none'
        # )
    else:
        pass
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.use_mixup or args.use_cutmix:
        train_loss_func = SoftCrossEntropyLoss(reduction='mean')
    else:
        train_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    test_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # train_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    # test_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    
    if args.use_sam:
        batch_updater = sam_batch_updater
    elif args.use_eisam:
        batch_updater = eisam_batch_updater
    else:
        batch_updater = basic_batch_updater

    start_epoch = 0
    best_acc = 0.0
    logs = []
    logs_array = np.zeros((args.epochs - start_epoch, 4))

    with tqdm(total=args.epochs, colour='MAGENTA', ascii=True) as pbar:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_loss, train_acc = trainer(net, optimizer, batch_updater, train_loader, train_loss_func, device)
            current_lr = optimizer.param_groups[0]['lr']
            current_rho = optimizer.param_groups[0].get('rho', None)
            if args.use_eisam:
                current_s = optimizer.param_groups[0].get('s', None)
            else:
                current_s = None
            # print(f"Epoch {epoch}/{args.epochs}, Learning Rate: {current_lr:.6f}")

            if epoch % args.log_interval == 0:
                test_loss, test_acc = validate(net, test_loader, test_loss_func, device)
                train_loss_scalar = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
                train_acc_scalar = train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
                test_loss_scalar = test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
                test_acc_scalar = test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
                logs_array[epoch - start_epoch - 1] = [train_acc_scalar, test_acc_scalar, train_loss_scalar, test_loss_scalar]

                if best_acc <= test_acc_scalar:
                    best_acc = test_acc_scalar

                if args.wandb:
                    log_dict = {
                        'train_acc': train_acc_scalar * 100,
                        'test_acc': test_acc_scalar * 100,
                        'train_loss': train_loss_scalar,
                        'test_loss': test_loss_scalar,
                        'best_acc': best_acc * 100,
                        'lr': current_lr,
                    }

                    if current_rho is not None:
                        log_dict['rho'] = current_rho
                    if current_s is not None:
                        log_dict['s'] = current_s
                    wandb_run.log(log_dict)


                message = (f'epoch: {epoch} lr: {current_lr:.6f} '
                           f'train_loss: {Fore.RED}{train_loss_scalar:.4f}{Style.RESET_ALL} '
                           f'train_acc: {Fore.RED}{train_acc_scalar*100:.2f}%{Style.RESET_ALL} '
                           f'test_loss: {Fore.GREEN}{test_loss_scalar:.4f}{Style.RESET_ALL} '
                           f'test_acc: {Fore.GREEN}{test_acc_scalar*100:.2f}%{Style.RESET_ALL} '
                           f'best_acc: {Fore.MAGENTA}{best_acc*100:.2f}%{Style.RESET_ALL}')
                pbar.set_description(message)
                pbar.update()
            if rho_scheduler is not None:
                rho_scheduler.step()

            if s_scheduler is not None:
                s_scheduler.step()
            scheduler.step()

        np.save(join(save_dir, 'logs.npy'), logs_array)
        torch.save(torch.from_numpy(logs_array), join(save_dir, 'logs.pt'))

# Trainer function
def trainer(net, optimizer, batch_updater, train_loader, train_loss_func, device):
    net.train()
    tot_loss = 0.0
    tot_correct = 0.0
    for _, batch in BackgroundGenerator(enumerate(train_loader)):
        loss, correct = batch_updater(net, optimizer, batch, train_loss_func, device)
        tot_loss += loss / len(train_loader.dataset)
        tot_correct += correct / len(train_loader.dataset)
    train_loss = tot_loss.item() if isinstance(tot_loss, torch.Tensor) else tot_loss
    train_acc = tot_correct.item() if isinstance(tot_correct, torch.Tensor) else tot_correct
    return train_loss, train_acc

# Validate function
def validate(net, test_loader, test_loss_func, device):
    net.eval()
    tot_loss = 0.0
    tot_correct = 0.0
    with torch.no_grad():
        for data, target in BackgroundGenerator(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            tot_loss += test_loss_func(output, target) * len(output)
            pred = output.argmax(dim=1, keepdim=True)
            tot_correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = (tot_loss.item() / len(test_loader.dataset)) if isinstance(tot_loss, torch.Tensor) else (tot_loss / len(test_loader.dataset))
    test_acc = tot_correct / len(test_loader.dataset)
    return test_loss, test_acc

def eisam_batch_updater(net, optimizer, batch, train_loss_func, device):
    data, target = batch
    data = data.to(device)

    if isinstance(target, (list, tuple)) and len(target) == 3:
        target1, target2, lam = target
        target1 = target1.to(device)
        target2 = target2.to(device)
        target_for_loss = (target1, target2, lam)   
        is_mixed = True
    else:
        target = target.to(device)
        target_for_loss = target
        is_mixed = False

    optimizer.zero_grad()
    enable_running_stats(net)
    output = net(data)
    loss = train_loss_func(output, target_for_loss)
    loss.backward()                         

    def closure():
        optimizer.zero_grad()
        output_pert = net(data)
        loss_pert = train_loss_func(output_pert, target_for_loss)
        loss_pert.backward()
        return loss_pert

    disable_running_stats(net)
    optimizer.step(closure)
    enable_running_stats(net)

    with torch.no_grad():
        output = net(data)
        loss = train_loss_func(output, target_for_loss)
        
        _, preds = torch.max(output.data, 1)
        if is_mixed:
            correct1 = preds.eq(target1).sum().item()
            correct2 = preds.eq(target2).sum().item()
            correct = lam * correct1 + (1 - lam) * correct2
        else:
            correct = preds.eq(target).sum().item()

    return loss.item() * len(data), correct

def basic_batch_updater(net, optimizer, batch, train_loss_func, device):
    optimizer.zero_grad()
    data, target = batch
    if not isinstance(target, (tuple, list)):
        target = target.to(device)        
    output = net(data.to(device))
    loss = train_loss_func(output, target)
    loss.backward()
    optimizer.step()    
    _, preds = torch.max(output.data, 1)
    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1, targets2 = targets1.to(device), targets2.to(device)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target).sum().item()
    
    return loss.item() * len(data), correct

def sam_batch_updater(net, optimizer, batch, train_loss_func, device):
    optimizer.zero_grad()
    data, target = batch
    data = data.to(device)
    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        target_device = (targets1, targets2, lam)
    else:
        target_device = target.to(device)
    
    def closure():
        loss = train_loss_func(net(data), target_device)
        loss.backward()
        return loss.item()
    enable_running_stats(net)    
    output = net(data)
    loss = train_loss_func(output, target_device)
    loss.backward()  

    disable_running_stats(net)
    optimizer.step(closure)     
    
    enable_running_stats(net)
    _, preds = torch.max(output.data, 1)
    
    if isinstance(target, (tuple, list)):
        targets1, targets2, _ = target_device
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target_device).sum().item()
    return loss.item() * len(data), correct

if __name__ == '__main__':
    main()