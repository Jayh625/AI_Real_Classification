import torch
import torch.nn as nn
import timm

class ViTBinaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTBinaryClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Identity() 
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)
        return x
