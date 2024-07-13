from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# https://github.com/pytorch/pytorch/issues/15849


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class SuperFastDataLoader():
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.q = mp.Queue()
        self.known_len = None
        self.loader: Optional[DataLoader] = None

        self.proc = mp.Process(target=self._proc)
        self.proc.start()
        self.known_len = self.q.get()

    def _proc(self):
        self.loader = DataLoader(*self.args, **self.kwargs)
        print(self.args, self.kwargs)
        print("Init loader in other proc", self.known_len)
        self.q.put(len(self.loader))
        for x in self.loader:
            print("Putting data")
            self.q.put(x)
        self.q.put(None)

    def __len__(self):
        return self.known_len

    def __iter__(self):
        while True:
            print("Wait for data ", self.known_len)
            x = self.q.get()
            if x is None:
                break
            yield x


class DummyInputGenerator:
    def __init__(self, batch_size, length=30, input_shape=224, num_class=100):
        self.batch_size = batch_size
        self.length = length
        self.a = torch.rand((self.batch_size, 3, input_shape, input_shape), device='cuda')
        self.b = torch.rand((self.batch_size, num_class), device='cuda')

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            yield self.a, self.b


from torchvision.datasets import ImageNet, CIFAR100, CIFAR10
from .datasets import ImageNet100
import torchvision.transforms as transforms

def get_sampler(dataset, world_size, rank):
    if world_size is not None and rank is not None:
        print(f"DistributedSampler world_size={world_size} rank={rank}")
        return DistributedSampler(dataset, world_size, rank)
    return RandomSampler(dataset)

def build_imagenet_dataloader(imagenet_dir: str, batch_size: int, num_workers: int, pin_memory: bool = True, split: str = "train", collate_fn = None, world_size = None, rank = None, worker_init_fn = None) -> FastDataLoader:

    
    
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    if not hasattr(build_imagenet_dataloader, "datasets"):
        build_imagenet_dataloader.datasets = {}
    if imagenet_dir not in build_imagenet_dataloader.datasets:
        print("Initializing new dataset")
        build_imagenet_dataloader.datasets[(imagenet_dir, split)] = ImageNet(imagenet_dir, split=split, transform=data_transforms[split])
    
    if collate_fn is not None:
        dataset = build_imagenet_dataloader.datasets[(imagenet_dir, split)]
        train_sampler = get_sampler(dataset, world_size, rank)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn, worker_init_fn=worker_init_fn,
        )

    dataset = build_imagenet_dataloader.datasets[(imagenet_dir, split)]
    train_sampler = get_sampler(dataset, world_size, rank)
    return FastDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

def build_imagenet100_dataloader(imagenet_dir: str, batch_size: int, num_workers: int, pin_memory: bool = True, split: str = "train", collate_fn = None, world_size = None, rank = None, worker_init_fn = None) -> FastDataLoader:
    
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    if not hasattr(build_imagenet100_dataloader, "datasets"):
        build_imagenet100_dataloader.datasets = {}
    if imagenet_dir not in build_imagenet100_dataloader.datasets:
        print("Initializing new dataset")
        build_imagenet100_dataloader.datasets[(imagenet_dir, split)] = ImageNet100(imagenet_dir, split=split, transform=data_transforms[split])

    dataset = build_imagenet100_dataloader.datasets[(imagenet_dir, split)]
    train_sampler = get_sampler(dataset, world_size, rank)
    return FastDataLoader(build_imagenet100_dataloader.datasets[(imagenet_dir, split)], batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

def build_cifar100_dataloader(cifar_dir: str, batch_size: int, num_workers: int, pin_memory: bool = True, split: str = "train", collate_fn = None, world_size = None, rank = None, worker_init_fn = None) -> FastDataLoader:

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
        }

    if not hasattr(build_cifar100_dataloader, "datasets"):
        build_cifar100_dataloader.datasets = {}
    if cifar_dir not in build_cifar100_dataloader.datasets:
        print("Initializing new dataset")
        build_cifar100_dataloader.datasets[(cifar_dir, split)] = CIFAR100(cifar_dir, train=(split == "train"), transform=data_transforms[split], download=False)
        
    dataset = build_cifar100_dataloader.datasets[(cifar_dir, split)]
    train_sampler = get_sampler(dataset, world_size, rank)
    return FastDataLoader(build_cifar100_dataloader.datasets[(cifar_dir, split)], batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

def build_cifar10_dataloader(cifar_dir: str, batch_size: int, num_workers: int, pin_memory: bool = True, split: str = "train", collate_fn = None, world_size = None, rank = None, worker_init_fn = None) -> FastDataLoader:

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
        }

    if not hasattr(build_cifar10_dataloader, "datasets"):
        build_cifar10_dataloader.datasets = {}
    if cifar_dir not in build_cifar10_dataloader.datasets:
        print("Initializing new dataset")
        build_cifar10_dataloader.datasets[(cifar_dir, split)] = CIFAR10(cifar_dir, train=(split == "train"), transform=data_transforms[split], download=False)

        
    dataset = build_cifar100_dataloader.datasets[(cifar_dir, split)]
    train_sampler = get_sampler(dataset, world_size, rank)
    return FastDataLoader(build_cifar10_dataloader.datasets[(cifar_dir, split)], batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

def build_dummy_dataloader(cifar_dir: str, batch_size: int, num_workers: int, pin_memory: bool = True, num_classes: int = 100, split: str = "train", collate_fn = None, world_size = None, rank = None, worker_init_fn = None) -> FastDataLoader:
    return DummyInputGenerator(batch_size, 128, num_class=num_classes)