import torch
import torch.nn as nn
import torchvision
from resnest.torch import resnest50, resnest101, resnest200


# for pretrained
PATH = {
    'resnest50' : '',
    'resnest101': '',
    'resnest200': '',
    'resnest269': '',
}

in_features = {
    'resnest50' : 2048,
    'resnest101': 2048,
    'resnest200': 2048,
    'resnest269': 2048
}



def net(model_name, pretrained, is_local, NUM_CLASS):

    if model_name == 'resnest50':
        if pretrained and not is_local:
            model = resnest50(pretrained=True)
        else:
            model = resnest50(pretrained=False)
    elif model_name == 'resnest101':
        if pretrained and not is_local:
            model = resnest101(pretrained=True)
        else:
            model = resnest101(pretrained=False)
    elif model_name == 'resnest200':
        if pretrained and not is_local:
            model = resnest200(pretrained=True)
        else:
            model = resnest200(pretrained=False)
    else:
        print('Error model name')

    if is_local:
        model_path = PATH[model_name]
        model.load_state_dict(torch.load(model_path), strict=False)

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features[model_name], NUM_CLASS)
    )

    return model
