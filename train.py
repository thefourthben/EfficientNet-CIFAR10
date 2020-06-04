import torch as t
from torch import nn
from torch.autograd import Variable
from torchnet import meter
import torchvision.datasets as tv
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Defining global variables
bacc = 0
n_epoch = 30
loss = t.nn.CrossEntropyLoss()


class training():
    def __init__(self):
        self.lr = 0.01
        self.effnet_type = "efficientnet-b0"
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
        self.trainset = None
        self.trainloader = None
        self.valset = None
        self.valloader = None
        self.testset = None
        self.testloader = None

    def preproc(self):
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = tv.CIFAR10(root='./data', train=True,
                            download=True, transform=transform_train)
        length = int(len(trainset) * 0.9)
        trainset, valset = t.utils.data.random_split(trainset, [length, len(trainset) - length])
        trainloader = t.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
        valloader = t.utils.data.DataLoader(valset, batch_size= 32, shuffle=True, num_workers=1)
        testset = tv.CIFAR10(root='./data', train=False,
                            download=True, transform=transform_train)
        testloader = t.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=True, num_workers=2)

    def train(self):
        # Getting the Model
        model = eff(effnet_type, len(classes))
        # Unfreezing the model parameters
        for param in model.parameters():
            param.requires_grad = True

        # Setting up the Loss function, optimizer, and channels
        optimizer = t.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.005)
        # optimizer = t.optim.RMSprop(model.parameters(), lr = init_lr, momentum=0.9, weight_decay=1e-5)
        # scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)
        # scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
        # setting the metric and confusion matrix
        lm = meter.AverageValueMeter()
        model.cuda()
        for num in range (30):
            model.train()
            lm.reset()
            total = 0
            correct = 0
            train_loss = 0 
            for i, (data, label) in enumerate(trainloader):
                input = Variable(data).cuda()
                target = Variable(label).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(input)
                l = loss(outputs, target)
                train_loss = l
                l.backward()
                optimizer.step()
                lm.add(l.data.item())
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            val_acc, val_loss= val(model, valloader)
            print('Epoch: {epoch} | Train Loss: {loss} | Train Acc: {train_acc} | Val Loss: {val_loss} |Val Acc: {val_acc}'.format(
                epoch = num, 
                loss = lm.value()[0],
                train_acc = 100 * correct/total,
                val_acc = val_acc,
                val_loss = val_loss, 
            ) )
            scheduler.step(val_acc)
        final_acc, _= self.val(model, testloader)
        print("Finished Training! The result is:{res}".format(
            res = final_acc
        ))



    def val(self, model, dataloader):
        model.cuda()
        model.eval()
        correct = 0
        total = 0
        l = 0
        with t.no_grad():
            for n, (inputs, label) in enumerate(dataloader):
                inputs, target = Variable(inputs).cuda(), Variable(label.long()).cuda()
                score = model(inputs)
                l = loss(score, target)
                _, predicted = t.max(score.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        # Save checkpoint.
        # acc = 100.*correct/total
        # if acc > bacc:
        #     bacc = acc
        #     print("New Best Accurarcy:" + str(bacc))
        return (100 * correct/total), l