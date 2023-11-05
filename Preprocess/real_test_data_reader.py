from lib2to3.pytree import type_repr
from pathlib import Path
import numpy as np
from PIL import Image
import random
import os

base_dir = Path("/datadrive/test")
folder = base_dir/"iid_test_split/context"
# folder = base_dir/"ood_test_split/context"
filepaths = sorted(list(folder.glob("**/*.npz")))

valid_idx = []

for idx in range(len(filepaths)):
    if idx % 10 == 0:
        print(idx, "/", len(filepaths))
        print("num of valid idx", len(valid_idx))
    
    filepath = filepaths[idx]
    npz = np.load(filepath)
        
    non_cloud_cnt = 0
    
    for t in range(10):

        t_img = npz["highresdynamic"][:,:,:,t].copy()
        # 'highresdynamic': [blue, green, red, nir, mask](128,128,5,X)
        t_blue, t_green, t_red, t_nir,t_cloud_mask = np.split(t_img, 5, axis=2)
        nan_mask = np.logical_or(np.isnan(t_red),np.logical_or(np.isnan(t_blue), np.isnan(t_green)))
        t_cloud_mask = np.logical_or(t_cloud_mask, nan_mask)
        
        if np.count_nonzero(t_cloud_mask) == 0:
            non_cloud_cnt += 1


    
    if non_cloud_cnt > 4:
        
        # filter out no cloudy days
        if non_cloud_cnt < 10:
            
            valid_idx.append(str(filepath))
    
 

valid_idx=np.array(valid_idx)
np.save("/home/real_test_iid_index.npy", valid_idx)
# np.save("/home/real_test_ood_index.npy", valid_idx)
print("num of valid idx", len(valid_idx))

if os.path.isfile("/home/real_test_iid_index.npy"):
    valid_filepaths = np.load("/home/real_test_iid_index.npy") 
# if os.path.isfile("/home/real_test_ood_index.npy"):
#     valid_filepaths = np.load("/home/real_test_ood_index.npy")

for idx in range(len(valid_filepaths)):
    filepath = valid_filepaths[idx]
    npz = np.load(filepath)
    cloud_lst = []
            
    
    for t in range(10):
        
        t_img = npz["highresdynamic"][:,:,:,t].copy()
        t_blue, t_green, t_red, t_nir, t_cloud_mask = np.split(t_img, 5, axis=2)
        nan_mask = np.logical_or(np.isnan(t_red),np.logical_or(np.isnan(t_blue), np.isnan(t_green)))
        t_cloud_mask = np.logical_or(t_cloud_mask, nan_mask)
                            
        t_channels = np.expand_dims(np.concatenate((t_blue, t_green, t_red, t_nir, t_cloud_mask), axis = 2), axis = 3)
        
        if t == 0:
            test_data_all = t_channels
        else:
            test_data_all = np.concatenate((test_data_all, t_channels), axis = 3)
                        

    test_data_all = np.array(test_data_all, dtype=np.float32)                                       # test_data_all (128, 128, 5, 10)

    np.save("/datadrive/real_test_iid/"+str(os.path.basename(filepath)),test_data_all)
    # np.save("/datadrive/real_test_ood/"+str(os.path.basename(filepath)),test_data_all)
