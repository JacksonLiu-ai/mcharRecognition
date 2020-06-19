from __future__ import print_function, absolute_import

import os
import numpy as np

from PIL import Image, ImageFilter
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageData(Dataset):
    def __init__(self, path_list, label_list, transform, image_path):
        self.dataset = path_list
        self.label = label_list
        self.transform = transform
        self.image_path = image_path

    def __getitem__(self, item):
        img = self.dataset[item]
        label = self.label[item]

        img = read_image(os.path.join(self.image_path, img))

        if self.transform is not None:
            img = self.transform(img)
        
        lbl = label  + (5 - len(label)) * [10]

        return img, np.array(lbl[:5])

    def __len__(self):
        return len(self.dataset)


class TestImageData(Dataset):
    def __init__(self, path_list, transform, image_path):

        self.dataset = path_list
        self.transform = transform
        self.image_path = image_path


    def __getitem__(self, item):
        imgname = self.dataset[item]

        img = read_image(os.path.join(self.image_path, imgname))

        if self.transform is not None:
            img = self.transform(img)
            
        return img, imgname

    def __len__(self):
        return len(self.dataset)