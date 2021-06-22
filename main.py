import os
import argparse
import glog as logging

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

def args_parse():
    parser = argparse.ArgumentParser(description="Pytorch re-implement for classic neural network.")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint.")
    parser.add_argument("--lr", default= 0.01, help="Learning rate, default is 0.01.")

    return parser.parse_args()

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
