from torch import nn
import timm


class EffNetb4(nn.Module):
    def __init__(self, model_name, n_outputs, is_sigmoid):
        super(EffNetb4, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_outputs)
        )
        self.is_sigmoid = is_sigmoid
        
    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x