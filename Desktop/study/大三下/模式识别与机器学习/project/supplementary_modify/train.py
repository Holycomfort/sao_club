from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TxtLoader
import models
import os
import matplotlib.pyplot as plt
import numpy as np

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir="./dataset1/train", input_size=628, batch_size=36):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomApply([transforms.Compose([
                    transforms.RandomCrop([300,300]),
                    transforms.Resize(input_size)
                    ])], p=0.2),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomRotation(45)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.2),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    image_dataset_train = TxtLoader('./dataset1/train.txt', data_transforms['train'])
    image_dataset_valid = TxtLoader('./dataset1/valid.txt', data_transforms['valid'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20):

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        global lr
        lr = lr / (1 + (epoch // 30) * 5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about training
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.U_Net(img_ch=1, output_ch=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data()

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)