"""
all research code is always a mess, i didn't care about clean code or anything like that here
"""

import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
import pandas as pd

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


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
    
    parser.add_argument('--datatype', default='F16', type=str)

    parser.add_argument('--lr', default = 1e-3, type=float)

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

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                transform=test_transform,
                                                download=True)
                                

    mp.spawn(train, nprocs=args.gpus, args=(train_dataset, test_dataset, args,))



def train(gpu, train_dataset, test_dataset, args):
    #DISTRIBUTED
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)


    #SETUPS
    setrandom(20214229)
    filename = "trace/"+str(args.rings) + "RINGS_" + args.datatype
    ext = ".csv"


    #MODEL AND DATATYPE 
    model = torchvision.models.resnet50(pretrained=False)

    #default is float32
    
    if args.datatype=="F16":
        model.half()
    elif args.datatype=="BF16": 
        model.bfloat16()

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d): #for numerical stability reasons, otherwise occasional NaN
            layer.float()

    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])


    #HYPERPARAMETERS
    batch_size = 128
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
 

    
    #DATASETS                           
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu)
                                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size = batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    train_eval_set = torch.utils.data.Subset(train_dataset, [random.randint(0,len(train_dataset)-1) for i in range(len(test_dataset))])

    train_loader_evaluation = torch.utils.data.DataLoader(dataset=train_eval_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True
                                                    )

    total_step = len(train_loader)

    if gpu==0:
        with open(filename+ext, "w+") as f:
            print("Loss", file=f)
        open(filename+"_accuracies.txt", "w+").close()

    model.train()

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            if args.datatype=="F16":
                images = images.cuda(gpu, non_blocking=True).half()
            elif args.datatype=="BF16":
                images = images.cuda(gpu, non_blocking=True).bfloat16()
            else:
                images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
                with open(filename+ext, "a+") as f:
                    print("{}".format(loss.item()), file=f)
        scheduler.step()

        if gpu==0 and epoch%50==0:
            evaluation(model, gpu, epoch+1, test_loader, filename, "Test set", args)
            evaluation(model, gpu, epoch+1, train_loader_evaluation, filename,"Train set", args)



def evaluation(model, gpu, epoch, dataloader, filename, evalname, args):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            if args.datatype=="F16":
                images = images.cuda(gpu, non_blocking=True).half()
            elif args.datatype=="BF16":
                images = images.cuda(gpu, non_blocking=True).bfloat16()
            else:
                images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    model.train()
    with open(filename+"_accuracies.txt", "a+") as f:
        print("Epoch {}. {} accuracy = {}%".format(epoch, evalname, accuracy), file=f)  

if __name__ == '__main__':
    main()