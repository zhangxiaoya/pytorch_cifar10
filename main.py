import os
import argparse
import glog as logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.lenet import LeNet
from utils import progress_bar

def args_parse():
    parser = argparse.ArgumentParser(description="Pytorch re-implement for classic neural network.")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint.")
    parser.add_argument("--lr", default= 0.01, help="Learning rate, default is 0.01.")

    return parser.parse_args()

# globale variable
best_acc = 0
start_epoch = 0

# Training
def train(epoch, net, train_loader, optimizer, criterion, device):
    logging.info("\nEpoch: {:06d}".format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)                   #
        total += targets.size(0)                        #
        correct += predicted.eq(targets).sum().item()   #

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc: {:.3f} % ({:d}/{:d})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing
def test(epoch, net, test_loader, criterion, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":
    logging.info("Pytorch re-implement classic nerual networks")
    args = args_parse()

    resume = args.resume
    learning_rate = args.lr
    batch_size = 128

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform= transform_train)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform= transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers = 2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 100,  shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using {}".format(device))

    # define network

    net = LeNet()
    net = VGG("VGG19")
    net.to(device)
    
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    creterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch,net, train_dataloader, optimizer, creterion, device)
        test(epoch, net, test_dataloader, creterion, device)
        scheduler.step()
