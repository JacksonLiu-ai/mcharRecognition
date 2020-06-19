# -*- coding: utf-8 -*
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import pandas as pd

import torch
from PIL import Image, ImageFilter
import config
from torch.autograd import Variable
from torchvision import transforms
from utils import auto_augment

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config.Config()

class Prediction():

    def load_model(self):

        model1 = torch.load(cfg.model_path + '/' + "best1.pth")
        self.model1 = model1.to(device)
        print('Load model1 successful')
        model2 = torch.load(cfg.model_path + '/' + "best2.pth")
        self.model2 = model2.to(device)
        print('Load model2 successful')

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/BeijingGarbage\/image\/0.png"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0"}
        '''
        # print(image_path)
        img = Image.open(image_path).convert('RGB')
        
        test_trainsform = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.crop_image_size),
            auto_augment.AutoAugment(dataset=cfg.autoaugment_dataset),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # output = 0
        r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
        for _ in range(cfg.tta_times):
            tensor = test_trainsform(img)
            tensor = torch.unsqueeze(tensor, dim=0).float()

            v1, v2, v3, v4, v5 = self.model1(tensor.to(device))
            t1, t2, t3, t4, t5 = self.model2(tensor.to(device))
            # v1, v2, v3, v4, v5 = self.model1(tensor.to(device))
            r1 += v1
            r2 += v2
            r3 += v3
            r4 += v4
            r5 += v5
            r1 += t1
            r2 += t2
            r3 += t3
            r4 += t4
            r5 += t5
        
        val_pred1 = torch.max(r1, 1)[1]
        val_pred2 = torch.max(r2, 1)[1]
        val_pred3 = torch.max(r3, 1)[1]
        val_pred4 = torch.max(r4, 1)[1]
        val_pred5 = torch.max(r5, 1)[1]
   
        val_pred = torch.stack([val_pred1, val_pred2, val_pred3, val_pred4, val_pred5], 1).squeeze()
        pred = list(val_pred.cpu().numpy())
        pred = [i for i in pred if i != 10]
        temp = ''
        for v in pred:
            temp += str(v)
        return int(temp)


    def get_result(self):
        
        
        image_names = os.listdir(os.path.join(cfg.data_path, 'mchar_test_a'))
        image_names.sort()
        results = []
        for img_name in tqdm(image_names):
            results.append(self.predict(os.path.join(cfg.data_path, 'mchar_test_a', img_name)))

        values = np.array([image_names, results])
        values = values.transpose()
        res = pd.DataFrame(values, columns=['file_name', 'file_code'])
        res.to_csv(os.path.join(cfg.result_path, cfg.result_file+'.csv'), index=False)




if __name__ == "__main__":
    predictor = Prediction()
    predictor.load_model()
    predictor.get_result()
    
    


