import torch.nn as nn
import timm

class EffNetb7(nn.Module):
    def __init__(self, model_name, num_classes, is_sigmoid):
        super(EffNetb7, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=num_classes)
        self.is_sigmoid = is_sigmoid
    
    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x