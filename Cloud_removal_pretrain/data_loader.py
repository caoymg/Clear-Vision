import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
from pathlib import Path
import numpy as np
import re


class EarthNet2021Dataset(Dataset):
    def __init__(self, folder: Union[Path, str]):
        
        # the list of sorted file paths is assigned to the attribute self.filepaths. 
        # The purpose of this code seems to be collecting a list of file paths with the .npz extension from the specified directory (folder) 
        # and its subdirectories, while ensuring that the "target" and "context" subdirectories are not present in the folder.
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.filepaths = sorted(list(folder.glob("*.npy")))  
        
    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]
        npz = np.load(filepath)

        # （T, c, 128, 128）
        highresdynamic = np.transpose(npz,(3,2,0,1))
        
        images = highresdynamic[:,:3,:,:]
        masks = highresdynamic[:,4,:,:]
        

        # preprocess
        images = images*10000
        images[images > 10000] = 10000
        images[images < 0] = 0
        minmax_gap = np.max(images) - np.min(images)
        images = images/minmax_gap
        
        gt = images.copy()
        
        # mask the cloud with max intensity 1
        T = highresdynamic.shape[0]
        for idx in range(T):
            cloud_mask = masks[idx,:,:]
            images[idx,0,:,:][cloud_mask == 1] = 1
            images[idx,1,:,:][cloud_mask == 1] = 1
            images[idx,2,:,:][cloud_mask == 1] = 1
        
        # padding
        if T < 10:
            num_frames_to_pad = 10 - T
            for i in range(num_frames_to_pad):
                images = np.concatenate((images, np.expand_dims(np.zeros_like(images[0]),axis=0)), axis=0)
                gt = np.concatenate((gt, np.expand_dims(np.ones_like(images[0]),axis=0)*255), axis=0)
        

        data = {
            "images": torch.from_numpy(images),
            "gt": torch.from_numpy(gt),
            "T": torch.tensor(T, dtype=torch.int8),              
            "cubename": self.__name_getter(filepath)
        }
        
        return data
    
    def __len__(self) -> int:
        return len(self.filepaths)
    
    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        """       
        
        # returns the last component of the path (the file name) as a string. 
        components = path.name.split("_")
        # match strings with the pattern of two digits followed by three uppercase letters
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert(bool(regex.match(components[1])))
            return "_".join(components[1:]) 
        


