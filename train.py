import torch as t
from torch import nn 
from torch.autograd import Variable
from torchnet import meter
import torchvision.datasets as tv
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import efficientnet_pytorch as eff
import torch.nn 
import torch.nn.functional as F


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(epoch):
    # vis = Visualizer('default')

    # Getting the data
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = tv.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=4)

    testset = tv.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = t.utils.data.DataLoader(testset, batch_size=16,
                                            shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    # Getting the model
    model = eff.EfficientNet.from_pretrained("efficientnet-b0")

    # Unfreezing the model parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Setting up the Fully Connected Layer
    num_param = model._fc.in_features
    model._fc = nn.Linear(num_param, 10)

    # Setting up the Loss function, optimizer, and channels
    loss = t.nn.CrossEntropyLoss()
    lr = 0.256
    optimizer = t.optim.Adam(model.parameters(), lr = lr, weight_decay= 1e-4)

    # setting up the metric and confusion matrix
    lm = meter.AverageValueMeter()
    cm = meter.ConfusionMeter(2)
    previous_loss = 1e100
    loss_log = []

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (data, label) in enumerate(trainloader):
            input = Variable(data)
            target = Variable(label)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input)
            l = loss(outputs, target)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            if i % 500 == 499:
                print(i+1)
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Done")
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        

def val(model, dataloader):
    model.eval()
    cm = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile = True)
        val_label = Variable(label.long(), volatile = True)
        score = model(val_input)
        cm.add(score.data.squeeze(), label.long())

    model.train()
    cm_value = cm.value()
    acc = 100. * (cm_value[0][0] + cm_value[1][1] / cm_value.sum())
    return cm, acc



if __name__ == "__main__":
    train(5)