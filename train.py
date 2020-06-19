# -*- coding: utf-8 -*-
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

# from data_loader import ImageData
from models import resnet, efficientnet, resnest, resnext, mymodel
from utils import auto_augment, losses, data_loader
import pandas as pd
import config
import json
from tqdm import tqdm

cfg = config.Config()

if not os.path.exists(cfg.model_path):
    os.mkdir(cfg.model_path)

# whether use gpu
use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class Main():
    
    def data_process(self):
        normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if cfg.use_autoaugment:
            train_transform = transforms.Compose([
                transforms.Resize(cfg.image_size),
                transforms.CenterCrop(cfg.crop_image_size),
                # transforms.RandomRotation(cfg.rotate_degree),
                auto_augment.AutoAugment(dataset=cfg.autoaugment_dataset),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.6, 1.2)),
                normal
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(cfg.image_size),
                transforms.CenterCrop(cfg.crop_image_size),
                transforms.RandomRotation(cfg.rotate_degree),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.6, 1.2)),
                normal
            ])

        test_trainsform = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.crop_image_size),
            transforms.ToTensor(),
            normal
        ])

        return train_transform, test_trainsform


    def train_model(self, model, epochs, train_loader, val_loader, save_model_name):
        
        model.to(DEVICE)

        if cfg.use_label_smooth:
            criterion = losses.CrossEntropyLabelSmooth(cfg.num_class, epsilon=0.1)
        else:
            criterion = nn.CrossEntropyLoss()
        criterion.to(DEVICE)
        
        if cfg.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum_rate, weight_decay=cfg.weight_decay_rate, nesterov=cfg.use_nesterov)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for i, (img, label) in enumerate(train_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                # print(img.size())
                # print(label.size())

                optimizer.zero_grad()
                # output = model(img)
                c0, c1, c2, c3, c4 = model(img)
                loss = criterion(c0, label[:, 0]) + \
                        criterion(c1, label[:, 1]) + \
                        criterion(c2, label[:, 2]) + \
                        criterion(c3, label[:, 3]) + \
                        criterion(c4, label[:, 4])
                # loss = criteration(output, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("Epoch {}, Step {}, Train_Loss {:.4f}".format(epoch, i , train_loss))
                    train_loss = 0
            scheduler.step()
            if epoch % 1 == 0 or epoch == epochs - 1:
                correct = 0
                model.eval()
                for val_img, val_label in tqdm(val_loader):
                    val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)
                    p1, p2, p3, p4, p5 = model(val_img)
                    val_pred1 = torch.max(p1, 1)[1]
                    val_pred2 = torch.max(p2, 1)[1]
                    val_pred3 = torch.max(p3, 1)[1]
                    val_pred4 = torch.max(p4, 1)[1]
                    val_pred5 = torch.max(p5, 1)[1]
                    # print(val_pred1)
                    # print(val_pred2)
                    # print(val_pred3)
                    # print(val_pred4)
                    # print(val_pred5)
                    val_pred = torch.stack([val_pred1, val_pred2, val_pred3, val_pred4, val_pred5], 1)
                    for i in range(val_pred.size(0)):
                        flag = True
                        for j in range(val_pred[i].size(0)):
                            if val_pred[i][j].item() != val_label[i][j].item():
                                flag = False
                        if flag:
                                correct += 1

                print("Epoch {},  Accuracy {:.2f}%".format(epoch, 100 * correct / len(val_loader.dataset)))
                # torch.save(model, cfg.model_path + '/epoch_' + str(epoch) + '.pth')

        torch.save(model, cfg.model_path + '/' + save_model_name)
        print(f'Current model-{save_model_name} ends, saved the model.')


    def train(self):

        # df = pd.read_csv(os.path.join(cfg.data_path, 'train.csv'))
        # image_path_list = df['image_path'].values
        # label_list = df['label'].values

        # split dataset
        # all_size = len(image_path_list)
        # train_size = int(all_size * cfg.train_per)
        # train_image_path_list = image_path_list[:train_size]
        # train_label_list = label_list[:train_size]

        # val_image_path_list = image_path_list[train_size:]
        # val_label_list = label_list[train_size:]


        # print(
            # 'train_size: %d, val_size: %d' % (len(train_image_path_list), len(val_image_path_list)))
        train_image_path_list = os.listdir(os.path.join(cfg.data_path, 'mchar_train'))
        train_image_path_list.sort()
        train_image_path_list = [os.path.join('mchar_train', name) for name in train_image_path_list]
        with open(os.path.join(cfg.data_path, 'mchar_train.json')) as t_f:
            train_labels = json.load(t_f)
        train_label_list = [train_labels[img]['label'] for img in train_labels.keys()]

        val_image_path_list = os.listdir(os.path.join(cfg.data_path, 'mchar_val'))
        val_image_path_list.sort()
        val_image_path_list = [os.path.join('mchar_val', name) for name in val_image_path_list]
        with open(os.path.join(cfg.data_path, 'mchar_val.json')) as v_f:
            val_labels = json.load(v_f)
        val_label_list = [val_labels[img]['label'] for img in val_labels.keys()]


        train_transform, val_transform = self.data_process()

        train_data = data_loader.ImageData(train_image_path_list, train_label_list, train_transform, cfg.data_path)
        val_data = data_loader.ImageData(val_image_path_list, val_label_list, val_transform, cfg.data_path)

        # filename = '/home/data/14'
        # all_data = torchvision.datasets.ImageFolder(filename, transform=train_transform)
        # trainsize = int(0.9 * len(all_data))
        # valsize = len(all_data) - trainsize
        # train_data, val_data = torch.utils.data.random_split(all_data,[trainsize, valsize])
        # print(all_data.class_to_idx)

        # print(train_data.class_to_idx)
        train_loader = DataLoader(train_data, batch_size=cfg.train_batch_size, shuffle=True, num_works = cfg.num_worker, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=cfg.val_batch_size, shuffle=False, num_works = cfg.num_worker, drop_last=True)


        # model1 = mymodel.MyModel('resnest101', cfg.pretrained, cfg.is_local, cfg.num_class)
        # print(model1)
        # print('Loading model1...')
        # self.train_model(model1, cfg.epochs, train_loader, val_loader, 'best1.pth')

        model2 = mymodel.MyModel2('efficientnet-b6', cfg.pretrained, cfg.is_local, cfg.num_class)
        if torch.cuda.device_count() > 1:
            model2 = torch.nn.DataParallel(model2, device_ids=cfg.device_id)
        print(model2)
        # print('Loading model2...')
        self.train_model(model2, cfg.epochs, train_loader, val_loader, 'best2.pth')
    
         
if __name__ == '__main__':
    main = Main()
    main.train()


