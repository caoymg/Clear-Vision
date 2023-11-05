import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
from pathlib import Path
import numpy as np
import re


class CloudMaskDataset(Dataset):
    def __init__(self, folder: Union[Path, str], type):
        
        # the list of sorted file paths is assigned to the attribute self.filepaths. 
        # The purpose of this code seems to be collecting a list of file paths with the .npz extension
        # from the specified directory (folder) and its subdirectories.
        if not isinstance(folder, Path):
            folder = Path(folder)
        filepaths = sorted(list(folder.glob("*.npy")))

        for idx in range(len(filepaths)):
            if type == "test":
                if idx % 100 == 0:
                    print("constructing Testset, now at", idx, "/", len(filepaths))
            else:
                if idx % 100 == 0:
                    print("constructing Trainset, now at", idx, "/", len(filepaths))
            filepath = filepaths[idx]
            npz = np.load(filepath)
            # （T, c, 128, 128）
            highresdynamic = np.transpose(npz,(3,2,0,1))
            images = highresdynamic[:,:3,:,:]
            masks = highresdynamic[:,4,:,:]
            
            # preprocess
            images = images*10000
            images[images > 10000] = 10000
            images[images < 0] = 0
            images = np.nan_to_num(images, nan=10000)
            minmax_gap = np.max(images) - np.min(images)
            images = images/minmax_gap

            if idx == 0:
                all_images = images
                all_masks = masks
            else:
                all_images = np.concatenate((all_images, images), axis = 0)
                all_masks = np.concatenate((all_masks, masks), axis = 0)
            # cnt = 0
            # # find cloudy day
            # T = highresdynamic.shape[0]
            # for t in range(T):
            #     cloud_mask = masks[t,:,:]
            #     image = images[t,:,:,:]
            #     image = np.expand_dims(image, axis=0)
            #     mask = cloud_mask
            #     mask = np.expand_dims(mask, axis=0)
            #     if np.count_nonzero(cloud_mask) != 0:
            #         if init:
            #             init = 0
            #             all_images = image
            #             all_masks = mask
            #         else:
            #             all_images = np.concatenate((all_images, image), axis = 0)
            #             all_masks = np.concatenate((all_masks, mask), axis = 0)
            #     if np.count_nonzero(cloud_mask) == 16384:
            #         cnt += 1

            # if cnt > 0:
            #     T = highresdynamic.shape[0]
            #     for t in range(T):
            #         cloud_mask = masks[t,:,:]
            #         image = images[t,:,:,:]
            #         image = np.expand_dims(image, axis=0)
            #         mask = cloud_mask
            #         mask = np.expand_dims(mask, axis=0)
            #         if np.count_nonzero(cloud_mask) == 0:
            #             all_images = np.concatenate((all_images, image), axis = 0)
            #             all_masks = np.concatenate((all_masks, mask), axis = 0)
            #             break


        self.all_data = all_images
        self.all_target = all_masks

    def __getitem__(self, idx: int) -> dict:
        
        data = {
            "images": torch.from_numpy(self.all_data[idx]),
            "gt": torch.from_numpy(self.all_target[idx])
        }
        
        return data
    
    def __len__(self) -> int:
        return len(self.all_data)


# class CloudMaskDataset(Dataset):
#     def __init__(self, folder: Union[Path, str]):
        
#         # the list of sorted file paths is assigned to the attribute self.filepaths. 
#         # The purpose of this code seems to be collecting a list of file paths with the .npz extension
#         # from the specified directory (folder) and its subdirectories.
#         if not isinstance(folder, Path):
#             folder = Path(folder)
#         self.filepaths = sorted(list(folder.glob("*.npy")))
        
#     def __getitem__(self, idx: int) -> dict:
        
#         filepath = self.filepaths[idx]
#         npz = np.load(filepath)

#         # （T, c, 128, 128）
#         highresdynamic = np.transpose(npz,(3,2,0,1))
#         images = highresdynamic[:,:3,:,:]
#         masks = highresdynamic[:,4,:,:]
        
#         # preprocess
#         images = images*10000
#         images[images > 10000] = 10000
#         images[images < 0] = 0
        
#         images = np.nan_to_num(images, nan=10000)

#         minmax_gap = np.max(images) - np.min(images)
#         images = images/minmax_gap
                
#         # find one cloudy day
#         T = highresdynamic.shape[0]
#         for idx in range(T):
#             cloud_mask = masks[idx,:,:]
#             if np.count_nonzero(cloud_mask) != 0:
#                 image = images[idx,:,:,:]
#                 mask = cloud_mask

#         data = {
#             "images": torch.from_numpy(image),
#             "gt": torch.from_numpy(mask)
#         }
        
#         return data
    
#     def __len__(self) -> int:
#         return len(self.filepaths)