import os
from enum import Enum
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

#CLASSES = [0, 1, 2] # 0: pore, 1: c, 2: Ni
# enumeration class representing each material type
class Material(Enum):
    PORE = 0
    CARBON = 1
    NICKEL = 2

class BatteryDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        fids = os.listdir(images_dir)
        ids = []
        for image_id in fids:
            if image_id[-3:] != 'png':
                continue
            ids.append(image_id)
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.weight = None
        
    def __len__(self): 
        return len(self.ids)
            
    def __getitem__(self, i):
        # read data and convert to greyscale
        image = Image.open(self.images_fps[i]).convert('L')
        
        # convert image to numpy array
        img_ndarray = np.asarray(image)
        # add an extra dimension
        img_ndarray = img_ndarray[np.newaxis, ...]
        # divide by 255 to normalize to [0, 1]
        image = img_ndarray/255.0
        
        masks = Image.open(self.masks_fps[i]).convert('L')
        
        masks = np.array(masks, dtype=np.float32)
        masks = masks/255.0
        nmasks = np.where((masks > 0) & (masks < 1.0), 2.0, masks)
        masks = nmasks
        
        mask_shape = (len(Material), masks.shape[0], masks.shape[1])
        new_masks = np.zeros(mask_shape)
        
        for c in range(len(Material)):
            ix_list = np.argwhere(masks==Material(c).value)
            for pos in ix_list:
                new_masks[c, pos[0], pos[1]] = 1
        
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'one_hot_mask': torch.as_tensor(new_masks.copy()),
            'true_mask': torch.as_tensor(masks.copy()),
            'id' : self.ids[i]
        }

