from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dataset import TxtLoader
from my_loss import SoftDiceLoss, BCEFocalLoss, lovasz_hinge, binary_xloss
import models, models2
import os
import matplotlib.pyplot as plt
import numpy as np


## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir="./dataset1/train", input_size=672, batch_size=4):
    data_transforms = {
        'train_all': transforms.Compose([
            #transforms.RandomCrop([300, 300]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomApply([transforms.RandomRotation(45)], p=0.3),
            transforms.Resize(input_size),
        ]),
        'train_data': transforms.Compose([
            #transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.3),
            transforms.ToTensor()
        ]),
        'valid_all': transforms.Compose([
            transforms.Resize(input_size),
        ]),
        'valid_data': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    image_dataset_train = TxtLoader('./dataset1/train.txt', data_transforms['train_all'], data_transforms['train_data'])
    image_dataset_valid = TxtLoader('./dataset1/valid.txt', data_transforms['valid_all'], data_transforms['valid_data'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader


def train_model(model, train_loader, valid_loader, criterion, dice, optimizer, num_epochs=20):

    '''
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        global lr
        lr = lr / (1 + (epoch // 10) * 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    '''

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_in, total_un = 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = binary_xloss(outputs[:,0,:,:], labels)# + dice(outputs, labels)
            otp = outputs.cpu().detach()
            predictions = np.ones_like(otp)
            predictions[otp <= 0.5] = 0
            predictions[otp > 0.5] = 1
            #_, predictions = torch.max(outputs, 1)
            #print(predictions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predictions, labels = np.array(predictions).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
            cell_in = np.sum(predictions * labels)
            cell_un = np.sum(predictions + labels)

            total_in += cell_in
            total_un += cell_un

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_iou = total_in / total_un
        return epoch_loss, epoch_iou

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_in, total_un = 0, 0

        with torch.no_grad():

            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = model(inputs)
                loss = 0
                otp = outputs.cpu().detach()
                predictions = np.ones_like(otp)
                predictions[otp <= 0.5] = 0
                predictions[otp > 0.5] = 1
                #_, predictions = torch.max(outputs, 1)

                total_loss += 0 * inputs.size(0)
                predictions, labels = np.array(predictions).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
                cell_in = np.sum(predictions * labels)
                cell_un = np.sum(predictions + labels)

                total_in += cell_in
                total_un += cell_un
            
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_iou = total_in / total_un
        return epoch_loss, epoch_iou

    best_iou = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_iou = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_iou))
        valid_loss, valid_iou = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_iou))
        scheduler.step()
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_model = model
            torch.save(best_model.state_dict(), 'best_model.pt')


def symmetric_lovasz(outputs, targets):
        return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    ## about training
    num_epochs = 100
    lr = 0.01

    ## model initialization
    model = nn.DataParallel(models2.U_Net(1, 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = load_data()

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    ## loss function
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.5]).to(device))
    criterion = nn.BCEWithLogitsLoss()
    #criterion = BCEFocalLoss()
    dice = SoftDiceLoss()
    train_model(model,train_loader, valid_loader, criterion, dice, optimizer, num_epochs=num_epochs)
