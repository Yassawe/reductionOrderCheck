"""
THE PROBLEM: I CANNOT ACCESS WEIGHTS BEFORE THEY ARE ALREADY REDUCED, AT WHICH POINT PARAMS IN ALL INSTANCES OF THE MODEL ARE THE SAME

"""
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
import pandas as pd


train_transform = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(p=.40),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def setrandom(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--rings', default=4, type=int, help='num of nccl rings')

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8008'

    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NCCL_MAX_NCHANNELS'] = str(args.rings)
    os.environ['NCCL_MIN_NCHANNELS'] = str(args.rings)
    os.environ['NCCL_DEBUG'] = "INFO"

    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=True,
                                               transform=train_transform,
                                               download=True)
    

    target_indecies = [random.randint(1, 50000000) for i in range(10)]


    mp.spawn(train, nprocs=args.gpus, args=(train_dataset, target_indecies, args,))



def train(gpu, train_dataset, target_indecies, args):

    setrandom(20214229)
 
    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)
    
    model = torchvision.models.resnet152(pretrained=False)

    torch.cuda.set_device(gpu)

    model.cuda(gpu)

    batch_size = 32

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9)
    LRSched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                                               
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.gpus,
                                                                    rank=gpu)
                                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    total_step = len(train_loader)

    target_iter = 5

    filename = "./weights/GPU" + str(gpu) + ".txt"

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(i)

        

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        if i == target_iter:
            g = torch.Tensor().cuda()
            for params in model.parameters():
                t = params.data
                t = torch.flatten(t)
                g = torch.cat((g,t))
            
            with open(filename, "a+") as f:
                for idx in target_indecies:
                    print("Index {} : Value {}".format(idx, g[idx].item()), file=f)
            
            break

        optimizer.step()


if __name__ == '__main__':
    main()