import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
from pathlib import Path
import numpy as np
import re
from PIL import Image

class PngData(Dataset):
    def __init__(self, folder: Union[Path, str]):
        
        # the list of sorted file paths is assigned to the attribute self.filepaths. 
        # The purpose of this code seems to be collecting a list of file paths with the .npz extension
        # from the specified directory (folder) and its subdirectories.
        if not isinstance(folder, Path):
            folder = Path(folder)
        filepaths = sorted(list(folder.glob("*.png")))
        
        for idx in range(len(filepaths)):

            image = Image.open(filepaths[idx]) 
            image = np.array(image)                             # image (128, 128, 3)
            
            r, g, b = np.split(image, 3, axis=2)
                                
            image = np.concatenate((b, g, r), axis = 2)          
            image = image.transpose(2,0,1)
            image = np.expand_dims(image, axis = 0)
             
            # PNG preprocess
            image = image/255
            
            if idx == 0:
                all_images = image
            else:
                all_images = np.concatenate((all_images, image), axis = 0)
                

        self.all_data = all_images.astype(np.float32)


    def __getitem__(self, idx: int) -> dict:
        
        data = {
            "images": torch.from_numpy(self.all_data[idx]),
        }
        
        return data
    
    def __len__(self) -> int:
        return len(self.all_data)



class NpyData(Dataset):
    def __init__(self, folder: Union[Path, str], file_idx):
        
        # the list of sorted file paths is assigned to the attribute self.filepaths. 
        # The purpose of this code seems to be collecting a list of file paths with the .npz extension
        # from the specified directory (folder) and its subdirectories.
        if not isinstance(folder, Path):
            folder = Path(folder)
        filepaths = sorted(list(folder.glob("*.npy")))
        
        filepath = filepaths[file_idx]
        npz = np.load(filepath)
        # （T, c, 128, 128）
        highresdynamic = np.transpose(npz,(3,2,0,1))
        images = highresdynamic[:,:3,:,:]
        
        # preprocess
        images = images
        images[images > 1] = 1
        images[images < 0] = 0
        images = np.nan_to_num(images, nan=1)
        minmax_gap = np.max(images) - np.min(images)
        images = images/minmax_gap

        all_images = images

        self.all_data = all_images


    def __getitem__(self, idx: int) -> dict:
        
        data = {
            "images": torch.from_numpy(self.all_data[idx]),
        }
        
        return data
    
    def __len__(self) -> int:
        return len(self.all_data)
    
    
    
class GradioData(Dataset):
    def __init__(self, input_imgs):
        
        # the list of sorted file paths is assigned to the attribute self.filepaths. 
        # The purpose of this code seems to be collecting a list of file paths with the .npz extension
        # from the specified directory (folder) and its subdirectories.
        
        for idx in range(len(input_imgs)):
            
            image = input_imgs[idx]
            r, g, b = np.split(image, 3, axis=2)
                                
            image = np.concatenate((b, g, r), axis = 2)          
            image = image.transpose(2,0,1)
            image = np.expand_dims(image, axis = 0)
             
            # PNG preprocess
            image = image/255
            
            if idx == 0:
                all_images = image
            else:
                all_images = np.concatenate((all_images, image), axis = 0)
                

        self.all_data = all_images.astype(np.float32)


    def __getitem__(self, idx: int) -> dict:
        
        data = {
            "images": torch.from_numpy(self.all_data[idx]),
        }
        
        return data
    
    def __len__(self) -> int:
        return len(self.all_data)