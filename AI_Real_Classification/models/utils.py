from models.effnet import EffNet
from models.effnetb4 import EffNetb4
from models.effnetb7 import EffNetb7
from models.vit import ViTBinaryClassifier
from models.resnet18 import ResNet18


def get_model(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet(**model_args)
    elif model_name == 'effnetb4' :
        return EffNetb4(**model_args)
    elif model_name == 'effnetb7' :
        return EffNetb7(**model_args)
    elif model_name == 'vit' :
        return ViTBinaryClassifier(**model_args)
    elif model_name == 'resnet18' :
        return ResNet18(**model_args)
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass