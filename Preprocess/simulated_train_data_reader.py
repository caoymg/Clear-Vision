from lib2to3.pytree import type_repr
from pathlib import Path
import numpy as np
from PIL import Image
import random
import os

base_dir = Path("/datadrive/train")
folder = base_dir/"train"
filepaths = sorted(list(folder.glob("**/*.npz")))

valid_idx = []

for idx in range(len(filepaths)):
    
    filepath = filepaths[idx]
    npz = np.load(filepath)
        
    t_cnt = 0
    
    for t in range(30):
        
        t_img = npz["highresdynamic"][:,:,:,t].copy()
        # 'highresdynamic': [blue, green, red, nir, cloud, scene, mask](128,128,7,X)
        t_blue, t_green, t_red, t_nir, _, _, t_cloud_mask = np.split(t_img, 7, axis=2)
        nan_mask = np.logical_or(np.isnan(t_red),np.logical_or(np.isnan(t_blue), np.isnan(t_green)))
        t_cloud_mask = np.logical_or(t_cloud_mask, nan_mask)
        
        if np.count_nonzero(t_cloud_mask) == 0:
            t_cnt += 1

    
    if t_cnt > 4:
        
        # filter out no cloudy days
        if t_cnt < 30:
        
            valid_idx.append(str(filepath))
            
        else:
            
            print("all cloudy~")
    
    if len(valid_idx)%1000 == 0:
        
        print("num of valid idx", len(valid_idx))
        
        
print("num of valid idx", len(valid_idx))

valid_idx=np.array(valid_idx)
np.save("/home/cloud_pixel_0.npy", valid_idx)

if os.path.isfile("/home/cloud_pixel_0.npy"):
    valid_filepaths = np.load("/home/cloud_pixel_0.npy")


# visualization
for idx in range(len(valid_filepaths)):
    filepath = valid_filepaths[idx]
    
    
    if os.path.isfile("/datadrive/processed_train/"+str(os.path.basename(filepath))+".npy"):
        
        print("cached file ", "/datadrive/processed_train/"+str(os.path.basename(filepath)))
    
    else:
        
        npz = np.load(filepath)
        
        cloud_lst = []
        
        first_flg = 1
        
        T = 0
        
        for t in range(30):
            
            t_img = npz["highresdynamic"][:,:,:,t].copy()
            # 'highresdynamic': [blue, green, red, nir, cloud, scene, mask](128,128,7,X)
            t_blue, t_green, t_red, t_nir, _, _, t_cloud_mask = np.split(t_img, 7, axis=2)
            nan_mask = np.logical_or(np.isnan(t_red),np.logical_or(np.isnan(t_blue), np.isnan(t_green)))
            t_cloud_mask = np.logical_or(t_cloud_mask, nan_mask)
            
            if np.count_nonzero(t_cloud_mask) == 0:
                
                if T < 10:
                    
                    t_channels = np.expand_dims(np.concatenate((t_blue, t_green, t_red, t_nir), axis = 2), axis = 3)
                    
                    if first_flg:
                        first_flg = 0
                        train_channels = t_channels
                    else:
                        train_channels = np.concatenate((train_channels, t_channels), axis = 3)
                    
                    T += 1
            
            else:
                
                cloud_lst.append(t)
                        
        # print("len(cloud_lst)", len(cloud_lst), "T", T)
        
        synthetic_num = random.randint(1,min(int(T/2), len(cloud_lst)))
        
        
        cloud_mask_list = []
        
        for idx in range(synthetic_num):
            
            cloud_mask = npz["highresdynamic"][:,:,6,cloud_lst[idx]].copy()
            cloud_mask_list.append(cloud_mask)
        
        for idx in range(T-synthetic_num):
            
            cloud_mask_list.append(np.zeros_like(cloud_mask))
            
        random.shuffle(cloud_mask_list)
        
            
        for t in range(T):
            
            train_channels_t = train_channels[:,:,:,t]
            cloud_mask_t = np.expand_dims(cloud_mask_list[t], axis=2)        
            train_data_t = np.concatenate((train_channels_t, cloud_mask_t), axis = 2)
            train_data_t = np.expand_dims(np.concatenate((train_channels_t, cloud_mask_t), axis = 2), axis = 3)
            
            if t == 0:
                
                train_data_all = train_data_t
                
            else:
                
                train_data_all = np.concatenate((train_data_all, train_data_t), axis = 3)
        
        
        np.save("/datadrive/processed_train/"+str(os.path.basename(filepath)),train_data_all)
                            
    
    