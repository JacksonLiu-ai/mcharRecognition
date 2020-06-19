import sys
import os

class Config():
    def __init__(self):
        self.num_class = 11

        self.use_distributed = True
        self.device_id = [0, 1]

        self.epochs = 25
        self.train_batch_size =  32
        self.val_batch_size = 32
        self.train_per = 0.96
        self.pretrained = True
        self.is_local = False
        self.num_worker = 4

        self.image_size = (100, 200)
        self.crop_image_size = (90, 180)
        self.rotate_degree = 5
        self.use_autoaugment = True
        self.autoaugment_dataset = 'CIFAR'

        self.use_label_smooth = True
        self.optim = 'sgd'
        self.loss_weight = [1] * self.num_class
        self.lr = 0.002
        self.momentum_rate = 0.9
        self.weight_decay_rate = 1e-4
        self.use_nesterov = True

        self.tta_times = 5
        self.result_path = os.path.join(sys.path[0], 'results')
        self.result_file = 'resnest101_resnext101'

        
        self.data_path = os.path.join(sys.path[0], 'data')
        self.model_path = os.path.join(sys.path[0], 'checkpoints')
