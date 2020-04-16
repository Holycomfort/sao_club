import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

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
            _, outputs = model(inputs)
            #print(outputs.shape, labels.shape)
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
        global valid_label, valid_pred
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
            valid_label.extend(labels.data)
            valid_pred.extend(predictions)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    global train_accs, valid_accs
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

    ## about model
    num_classes = 20

    ## about data
    data_dir = "../data/"
    inupt_size = 224
    batch_size = 36

    ## about training
    num_epochs = 3
    lr = 0.001
    train_accs = []
    valid_accs = []
    valid_label = []
    valid_pred = []    

    ## model initialization
    model = models.model_C(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=inupt_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)

    ## visualize
    t = plt.plot(train_accs)
    v = plt.plot(valid_accs)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(["train","valid"])
    plt.show()

    ## confusion matrix
    valid_label = list(map(int, valid_label))
    valid_pred = list(map(int, valid_pred))
    print(confusion_matrix(valid_label, valid_pred))
