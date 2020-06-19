import torch
import torch.nn as nn
import torchvision

from . import resnet, efficientnet, ghostnet, resnest, resnext


class MyModel(torch.nn.Module):
    def __init__(self, model_name, pretrained, is_local, NUM_CLASS):
        super().__init__()
        
        model = resnest.net(model_name, pretrained, is_local, NUM_CLASS)
        in_feature = NUM_CLASS

        # model_conv = nn.Sequential(*list(model.children())[:-1])

        self.conv = model
        
        self.fc1 = nn.Linear(in_feature, NUM_CLASS)
        self.fc2 = nn.Linear(in_feature, NUM_CLASS)
        self.fc3 = nn.Linear(in_feature, NUM_CLASS)
        self.fc4 = nn.Linear(in_feature, NUM_CLASS)
        self.fc5 = nn.Linear(in_feature, NUM_CLASS)

    def forward(self, img):      
        feat = self.conv(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5



class MyModel2(torch.nn.Module):
    def __init__(self, model_name, pretrained, is_local, NUM_CLASS):
        super().__init__()
        
        # model = resnext.net(model_name, pretrained, is_local, NUM_CLASS)
        model = efficientnet.net(model_name, pretrained, is_local, NUM_CLASS)

        in_feature = NUM_CLASS

        # model_conv = nn.Sequential(*list(model.children())[:-2])

        self.conv = model
        
        self.fc1 = nn.Linear(in_feature, NUM_CLASS)
        self.fc2 = nn.Linear(in_feature, NUM_CLASS)
        self.fc3 = nn.Linear(in_feature, NUM_CLASS)
        self.fc4 = nn.Linear(in_feature, NUM_CLASS)
        self.fc5 = nn.Linear(in_feature, NUM_CLASS)

    def forward(self, img):      
        feat = self.conv(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5