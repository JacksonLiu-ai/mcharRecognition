import torch
import torch.nn as nn
import torchvision

from efficientnet_pytorch import EfficientNet

# for local pretrained
PATH = {
    'efficientnet-b0' : '',
    'efficientnet-b1' : '',
    'efficientnet-b2' : '',
    'efficientnet-b3' : '',
    'efficientnet-b4' : '',
    'efficientnet-b5' : '',
    'efficientnet-b6' : '',
    'efficientnet-b7' : '',
}

in_features = {
    'efficientnet-b0' : 1280, 
    'efficientnet-b1' : 1280, 
    'efficientnet-b2' : 1408, 
    'efficientnet-b3' : 1536, 
    'efficientnet-b4' : 1792, 
    'efficientnet-b5' : 2048, 
    'efficientnet-b6' : 2304, 
    'efficientnet-b7' : 2560, 
}


def net(model_name, pretrained, is_local, NUM_CLASS):

    if pretrained and not is_local:
        model = EfficientNet.from_pretrained(model_name)
    else:

        model = EfficientNet.from_name(model_name)

        if is_local:

            model_path = PATH[model_name]
            model.load_state_dict(torch.load(model_path), strict=False)
            print('Load pretrained model successful')

    model._fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features[model_name], NUM_CLASS)
    )

    return model