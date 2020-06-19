import torch
import torch.nn as nn
import torchvision


# need to change!!!
PATH = {
    'resnet101': '',
    'resnet50': '',
    'resnet18': ''
}

in_features = {
    'resnet18' : 512,
    'resnet50' : 2048,
    'resnet101': 2048,
}

def net(model_name, pretrained, is_local, NUM_CLASS):


    if model_name == 'resnet101':
        if pretrained and not is_local:
            model = torchvision.models.resnet101(pretrained=True)
        else:
            model = torchvision.models.resnet101(pretrained=False)
    elif model_name == 'resnet50':
        if pretrained and not is_local:
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model = torchvision.models.resnet50(pretrained=False)
    elif model_name == 'resnet18':
        if pretrained and not is_local:
            model = torchvision.models.resnet18(pretrained=True)
        else:
            model = torchvision.models.resnet18(pretrained=False)
    else:
        print('Error model name')

    if is_local:
        model_path = PATH[model_name]
        model.load_state_dict(torch.load(model_path), strict=False)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features[model_name], NUM_CLASS)
    )

    return model
