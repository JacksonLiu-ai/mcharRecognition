import torch
import torch.nn as nn
import torchvision


# for pretrained
PATH = {
    'resnext50' : '',
    'resnext101': '',
}

in_features = {
    'resnext50' : 2048,
    'resnext101': 2048,
}

def net(model_name, pretrained, is_local, NUM_CLASS):


    if model_name == 'resnext50':
        if pretrained and not is_local:
            model = torchvision.models.resnext50_32x4d(pretrained=True)
        else:
            model = torchvision.models.resnext50_32x4d(pretrained=False)
    elif model_name == 'resnext101':
        if pretrained and not is_local:
            model = torchvision.models.resnext101_32x8d(pretrained=True)
        else:
            model = torchvision.models.resnext101_32x8d(pretrained=False)
    else:
        print('Error model name')

    if is_local:
        model_path = remote_helper.get_remote_date(PATH[model_name])
        model.load_state_dict(torch.load(model_path), strict=False)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features[model_name], NUM_CLASS)
    )

    return model
