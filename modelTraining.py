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


# train_transform = T.Compose([
#     T.Resize(224),
#     T.RandomHorizontalFlip(p=.40),
#     T.RandomRotation(30),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# test_transform = T.Compose([
#     T.Resize(224),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


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
    os.environ['MASTER_PORT'] = '1024'

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

    setrandom(20214229)
    dt = "F16"
    filename = "trace/"+str(args.rings) + "RINGS_" + dt
    ext = ".csv"

    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)
    

    model = torchvision.models.resnet50(pretrained=True)

    model.half()
    
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d): #for numerical stability reasons, otherwise occasional NaN
            layer.float()


    torch.cuda.set_device(gpu)

    model.cuda(gpu)

    batch_size = 128

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
 

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
                                               
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.gpus,
                                                                    rank=gpu)
                                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size = batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    # train_loader_eval = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                                 batch_size=batch_size,
    #                                                 shuffle=False,
    #                                                 pin_memory=True
    #                                                 )

    start = datetime.now()
    total_step = len(train_loader)
    
    if gpu==0:
        with open(filename+ext, "a+") as f:
            print("Loss", file=f)

    model.train()
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
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
        LRSched.step()

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    
    if gpu == 0:
        model.eval()
        # evaluating on the test set
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = 100 * correct / total
        
        # #evaluating on the train set
        # with torch.no_grad():
        #     correct = 0
        #     total = 0
        #     for images, labels in train_loader_eval:
        #         images = images.cuda(non_blocking=True).bfloat16()
        #         labels = labels.cuda(non_blocking=True)
        #         outputs = model(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        #     train_accuracy = 100 * correct / total
        
        with open(filename+"_accuracies.txt", "a+") as f:
            print("Test set accuracy = {}%".format(test_accuracy), file=f)  

if __name__ == '__main__':
    main()