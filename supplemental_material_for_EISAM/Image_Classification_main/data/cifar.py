import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms

from .cutout import Cutout
from .cutmix import CutMixCollator
from .mixup import MixUpCollator
#使用了数据增强
def get_cifar_dataloader(args):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    # basic_transform = [transforms.RandomCrop(32, padding=4),
    #                 transforms.RandomHorizontalFlip(),
    #                 ]
    basic_transform = [transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip()]

    if args.use_auto_augment:
        basic_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    if args.use_rand_augment:
        basic_transform.append(transforms.RandAugment(num_ops=2, magnitude=9))
    if args.use_random_erasing:
        basic_transform.append(transforms.RandomErasing(p=0.25))
        
    basic_transform += [transforms.PILToTensor(), 
                        transforms.ConvertImageDtype(torch.float), 
                        normalize]
    if args.use_cutout:
        basic_transform.append(Cutout(n_holes=1, length=args.length))
        
    train_transform = transforms.Compose(basic_transform)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    elif args.use_mixup:
        collator = MixUpCollator(args.mixup_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    if args.data_name == 'CIFAR-10':
        train_dataset = torchvision.datasets.CIFAR10(
            args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            args.data_dir, train=False, transform=test_transform, download=True)
    elif args.data_name == 'CIFAR-100':
        train_dataset = torchvision.datasets.CIFAR100(
            args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            args.data_dir, train=False, transform=test_transform, download=True)
    else:
        raise NotImplementedError(f"{args.data_name} is not avaiable right now !")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader