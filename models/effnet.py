import torch as t
from efficientnet_pytorch import model as m
from torch import nn

class eff(nn.Module):
    def __init__(self, backbone, out_dim):
        super(eff, self).__init__()
        self.effnet = m.EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.effnet._fc.in_features, out_dim)
        self.effnet._fc = nn.Identity()

    def extract(self, x):
        return self.effnet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
    def savemodel(self, val, model):
        t.save(model.state_dict(), 'checkpoint.pt')