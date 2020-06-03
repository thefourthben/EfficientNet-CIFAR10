import torch as t
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import torchvision.datasets as tv
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from efficientnet_pytorch import model as m
import torch.nn.functional as F

# Defining global variables
bacc = 0
n_epoch = 30


class eff(nn.Module):
    def __init__(self, backbone, out_dim):
        super(eff, self).__init__()
        self.effnet = m.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.effnet._fc.in_features, out_dim)
        self.effnet._fc = nn.Identity()

    def extract(self, x):
        return self.effnet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

class VGG(nn.Module):
    def __init__(self, out_dim):
        super(VGG, self).__init__()
        self.vgg = models.vgg16()
        self.myfc = nn.Linear(self.vgg._fc.in_features, out_dim)
        self.vgg._fc = nn.Identity()
    
    def extract(self, x):
        return self.effnet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

def train():
    # Setting the parameters
    init_lr = 0.256
    effnet_type = "efficientnet-b0"
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Getting the data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = tv.CIFAR10(root='./data', train=True,
                          download=True, transform=transform_train)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=4)

    testset = tv.CIFAR10(root='./data', train=False,
                         download=True, transform=transform_test)
    testloader = t.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=4)
    # Getting the Model
    model = eff(effnet_type, len(classes))

    # Unfreezing the model parameters
    for param in model.parameters():
        param.requires_grad = True

    # Setting up the Loss function, optimizer, and channels
    loss = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-4)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)

    # setting u   the metric and confusion matrix
    lm = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(2)
    previous_loss = 1e100
    for _ in range (n_epoch):
        total = 0
        correct = 0
        model.train()
        train_loss = 0 
        for i, (data, label) in enumerate(trainloader):
            if i == 0:
                print(i)
            input = Variable(data)
            target = Variable(label)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(input)
            l = loss(outputs, target)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(target).sum().item()
            if i % 500 == 499:
                print(i+1)
            if i % 1000 == 999:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (train_loss/(i+1), 100.*correct/total, correct, total))
        print("Done")
        model.save()
        val(model, testloader)
        scheduler.step()



def val(model, dataloader):
    global bacc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num = 0
    with torch.no_grad():
        for n, (inputs, label) in enumerate(testloader):
            if num == 0:
                print("yes it's working be patient")
                num += 1
            inputs, targets = Variable(input, volatile = True), Variable(label.long(), volatile = True)
            outputs = model(inputs)
            l = loss(outputs, targets)
            test_loss += l.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Test: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > bacc:
        bacc = acc
        print("New Best Accurarcy:" + bacc)

if __name__ == "__main__":
    train()    