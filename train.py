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
from models.effnet import EfficientNet as eff
from earlystop import EarlyStopping as early 
import torch.onnx
import onnx
import onnxruntime
from PIL import Image
import collections
from models import utils


class training():
    def __init__(self, model):
        self.lr = 0.01
        self.trainset = None
        self.trainloader = None
        self.valset = None
        self.valloader = None
        self.testset = None
        self.testloader = None
        self.whichmodel = model
        self.bacc = 0
        self.n_epoch = 3
        self.loss = t.nn.CrossEntropyLoss()
        self.factor = 0.5
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')        utils.BlockArgs
        self.block = [ 
            utils.BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
            utils.BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            utils.BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
            ]
        self.globalparm = utils.GlobalParams(
            width_coefficient=None,
            depth_coefficient=None,
            image_size=None,
            dropout_rate=0.2,

            num_classes=10,
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            drop_connect_rate=0.2,
            depth_divisor=8,
            min_depth=None
            )

    def preproc(self):
        transform_train = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Loading and getting the trainset
        self.trainset = tv.CIFAR10(root='./data', train=True,
                            download=True, transform=transform_train)
        length = int(len(self.trainset) * 0.9)
        self.trainset, self.valset = t.utils.data.random_split(self.trainset, [length, len(self.trainset) - length])
        self.trainloader = t.utils.data.DataLoader(self.trainset, batch_size=80,
                                            shuffle=True, num_workers=2)
        
        # Creating validation loader
        self.valloader = t.utils.data.DataLoader(self.valset, batch_size= 32, shuffle=True, num_workers=2)

        # Creating and loading the test set
        self.testset = tv.CIFAR10(root='./data', train=False,
                            download=True, transform=transform_train)
        self.testloader = t.utils.data.DataLoader(self.testset, batch_size=80,
                                            shuffle=True, num_workers=2)

    def train(self):
        self.preproc()
        
        # Getting the Model
        model = self.whichmodel(blocks_args=self.block, global_params=self.globalparm)
        # Unfreezing the model parameters
        for param in model.parameters():
            param.requires_grad = True

        # Setting up the Loss function, optimizer, and channels
        optimizer = t.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
        # optimizer = t.optim.RMSprop(model.parameters(), lr = init_lr, momentum=0.9, weight_decay=1e-5)
        # optimizer = t.optim.Adam(model.parameters(), lr = self.lr, weight_decay=0.000125)
        # scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epoch)
        # scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = "max",factor=self.factor,
         patience=5, verbose=True)
        # setting the metric and confusion matrix
        lm = meter.AverageValueMeter()
        model.cuda()
        for num in range (self.n_epoch):
            model.train()
            lm.reset()
            total = 0
            correct = 0
            train_loss = 0 
            for i, (data, label) in enumerate(self.trainloader):
                inputs = Variable(data).cuda()
                target = Variable(label).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                l = self.loss(outputs, target)
                train_loss = l
                l.backward()
                optimizer.step()
                lm.add(l.data.item())
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            val_acc, val_loss = self.val(model, self.valloader)
            print('Epoch: {epoch} | Train Loss: {loss} | Train Acc: {train_acc} | Val Loss: {val_loss} |Val Acc: {val_acc}'.format(
                epoch = num, 
                loss = lm.value()[0],
                train_acc = 100 * correct/total,
                val_acc = val_acc,
                val_loss = val_loss, 
            ) )
            scheduler.step(val_acc)
            early(val_acc, model)
        final_acc, _= self.val(model, self.testloader)
        print("Finished Training! The result is: {res}".format(
            res = final_acc
        ))
        model.savemodel(self.lr, 128, self.factor ,"Adam", "Plat", self.n_epoch)
        w = t.randn(1, 3, 224, 224).cuda()
        model.effnet.set_swish(memory_efficient=False)
        t.onnx.export(model, w, "result.onnx", verbose= False)
        self.onnxconversion()


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
                l = self.loss(score, target)
                _, predicted = t.max(score.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        # Save checkpoint.
        # acc = 100.*correct/total
        # if acc > bacc:
        #     bacc = acc
        #     print("New Best Accurarcy:" + str(bacc))
        return (100 * correct/total), l
    
    def onnxconversion(self):
        img = Image.open('airplane.jpg')
        transform_train = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])        
        img = transform_train(img)
        img = img.view(1, 3, 224, 224)
        onnx_model =  onnx.load("result.onnx")
        onnx.checker.check_model(onnx_model)
        ort = onnxruntime.InferenceSession("result.onnx")
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()
        # compute ONNX Runtime output prediction
        ort_inputs = {ort.get_inputs()[0].name: to_numpy(img)}
        t0 = time.clock()
        ort_outs = ort.run(None, ort_inputs)
        t1 = time.clock() - t0
        result = list(self.classes)
        print("Predicted result: {pres} | Actual Result: {ares} | Time Spent: {ts}".format(
                ares=result[0], 
                pres=result[ort_outs.index(max(ort_outs))], 
                ts=t1
        ))
        

        # compare ONNX Runtime and PyTorch results
        # print("airplane", )

if __name__ == "__main__":
    tr = training(eff)
    tr.train()    