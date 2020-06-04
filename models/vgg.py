import torch as t
from torch import nn

class VGG(nn.Module):
    def __init__(self, out_dim):
        super(VGG, self).__init__()
        self.vgg = models.vgg16()
        self.myfc = nn.Linear(self.vgg._fc.in_features, out_dim)
        self.vgg._fc = nn.Identity()
    
    def extract(self, x):
        return self.vgg(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
    def savemodel(self, val, model):
        t.save(model.state_dict(), 'checkpoint.pt')