import sys
import os
import torch
import torchvision
import torch.nn.functional as F
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../build"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../src/python/fasterdp"))
import fasterdp
from model_wrapper import FasterDpModelWrapper, StellaTrainDataLoader
from torch import multiprocessing as mp
from tqdm import tqdm

def test_end_to_end(local_rank: int, local_master_pid: int, local_world_size: int, args):

    ddp_rank = int(os.environ['RANK']) * local_world_size + local_rank
    ddp_world_size = int(os.environ['WORLD_SIZE']) * local_world_size
    print(ddp_world_size, ddp_rank)

    compress_option = "thresholdv16"

    # configure
    fasterdp.configure(
        os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else '127.0.0.1',
        int(os.environ['MASTER_PORT']) if 'MASTER_PORT' in os.environ else 5555,
        int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1,
        int(os.environ['RANK']) if 'RANK' in os.environ else 0,
        local_master_pid,
        local_world_size,
        local_rank,
        compress_option,
        args.gradient_accumulation
    )
    torch.cuda.set_device(torch.cuda.device(local_rank))

    fasterdp.configure_compression_ratio(args.compression)

    model_original = getattr(torchvision.models, args.model)(num_classes=args.num_classes)
    print(args)

    model_original = model_original.to('cuda')
    model:torch.nn.Module = FasterDpModelWrapper(model_original)
    model.train()
    model.to('cuda')

    # optimizer comfiguration
    lr = args.lr / 256. * ddp_world_size * args.batch_size 
    fasterdp.set_optimizer(args.optim)
    fasterdp.configure_optimizer('lr', lr)
    fasterdp.configure_optimizer('momentum', args.momentum)
    fasterdp.configure_optimizer('smart_momentum', False)

    global_step = 0

    fasterdp.barrier()
    for epoch in range (args.num_epochs):
        loader = args.dataloader_builder(args.dataset_root, args.batch_size, 6, world_size=ddp_world_size, rank=ddp_rank)
        if local_rank == 0:
            loader = tqdm(enumerate(loader), total=len(loader) if args.num_iters is None else args.num_iters)
        else:
            loader = enumerate(loader)

        my_loader = StellaTrainDataLoader(loader)
        for idx, (data, target) in my_loader:
            if idx == 0 and epoch == 0:
                print("Fed batch size is ", data.shape, "* gradacc", args.gradient_accumulation)

            if args.num_iters is not None and idx == args.num_iters:
                break

            data, target = data.cuda(), target.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()

            if my_loader.is_eoi:
                if local_rank == 0:
                    loader.set_postfix({
                        'loss': loss.cpu().item(),
                    })
                global_step += 1
        model.step()

    fasterdp.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model name')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='optimizer type')
    parser.add_argument('--policy', type=str, default='strided', choices=['sequential', 'strided', 'cache_friendly'],
                        help='compression policy type')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='number of classes')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='number of samples')
    parser.add_argument('--num-iters', type=int,
                        help='number of iterations')
    parser.add_argument('--batch-size', type=int, default=72,
                        help='per-device batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--compression', type=float, default=0.95,
                        help='compression 0-1')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--manual-spawn', type=int, default=None)
    parser.add_argument('--local-world-size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--local-master-pid', type=int, default=os.getpid())

    # optimizer configs
    parser.add_argument('--lr', type=float, default=0.00625)
    parser.add_argument('--momentum', type=float, default=0.9)

    # dataset configs
    parser.add_argument('--imagenet-root', type=str, default='/datasets/imagenet')
    parser.add_argument('--cifar100-root', type=str, default='/datasets/cifar100')
    parser.add_argument('--cifar10-root', type=str, default='/datasets/cifar10')
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet', 'imagenet100', 'cifar100', 'cifar10', 'dummy'])

    parser.add_argument('--gradient-accumulation', type=int, default=1)

    args = parser.parse_args()
    local_world_size = args.local_world_size

    print(f"Per-device batch size is {args.batch_size}")
    method = test_end_to_end

    if args.dataset == 'imagenet':
        from util.dataloader import build_imagenet_dataloader
        args.num_classes = 1000
        args.dataloader_builder = build_imagenet_dataloader
        args.dataset_root = args.imagenet_root
    elif args.dataset == 'imagenet100':
        from util.dataloader import build_imagenet100_dataloader
        args.num_classes = 100
        args.dataloader_builder = build_imagenet100_dataloader
        args.dataset_root = args.imagenet_root
    elif args.dataset == 'cifar100':
        from util.dataloader import build_cifar100_dataloader
        args.num_classes = 100
        args.dataloader_builder = build_cifar100_dataloader
        args.dataset_root = args.cifar100_root
    elif args.dataset == 'cifar10':
        from util.dataloader import build_cifar10_dataloader
        args.num_classes = 10
        args.dataloader_builder = build_cifar10_dataloader
        args.dataset_root = args.cifar10_root
    elif args.dataset == 'dummy':
        from util.dataloader import build_dummy_dataloader
        args.num_classes = 100
        args.dataloader_builder = build_dummy_dataloader
        args.dataset_root = args.imagenet_root

    if args.manual_spawn is not None:
        print(f"master PID is {args.local_master_pid}")
        method(args.manual_spawn, args.local_master_pid, local_world_size, args)
    else:
        mp.spawn(method,
                args=(os.getpid(), local_world_size, args),
                nprocs=local_world_size,
                join=True)


