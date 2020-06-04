from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dataset import TxtLoader
import models, models2
import os
import matplotlib.pyplot as plt
import numpy as np


## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir="./dataset1/train", input_size=628, batch_size=4):
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

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_in, total_un = 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + dice(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            #print(predictions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predictions, labels = np.array(predictions.cpu()).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
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
                loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)

                total_loss += loss.item() * inputs.size(0)
                predictions, labels = np.array(predictions.cpu()).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
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


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        probs = probs[:, 1, :, :]
        m1 = probs.view(num, -1)
        m2 = targets.float().view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.4, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)[:, 1, :, :].reshape(-1)
        tg = target.reshape(-1).float()
        alpha, gamma = self.alpha, self.gamma
        #print(pt.shape,target.shape)
        loss = - alpha * (1 - pt) ** gamma * tg * torch.log(pt+1e-10) - \
               (1 - alpha) * pt ** gamma * (1 - tg) * torch.log(1 - pt + 1e-10)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    ## about training
    num_epochs = 100
    lr = 0.01

    ## model initialization
    model = models2.U_Net(1, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = load_data()

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    ## loss function
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.5]).to(device))
    #criterion = BCEFocalLoss()
    dice = SoftDiceLoss()
    train_model(model,train_loader, valid_loader, criterion, dice, optimizer, num_epochs=num_epochs)
