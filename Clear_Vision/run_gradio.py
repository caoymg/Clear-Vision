import gradio as gr
import argparse
import torch
from networks.cloud_mask_model import *
from networks.cloud_removal_model import *
import os
from torch.utils import data
from data_loader import GradioData
from utils import generate_cloud_mask, cloud_removal


def remove_cloud(input_img_t0, input_img_t1, input_img_t2, input_img_t3, input_img_t4, input_img_t5, input_img_t6, input_img_t7, input_img_t8, input_img_t9, T):
    """
    input image (128, 128, 3)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_dir', type=str, default='./model_dict/',
                        help='where to save snapshots of the model.')
    torch.cuda.manual_seed_all(0)

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        cuda = 1
        print("CUDA is available on this system.")
    else:
        cuda = 0
        print("CUDA is not available on this system.")
    
    cloud_mask_classes = 2
    channels = 3
    cloud_mask_model = unet(cloud_mask_classes)
    cloud_removal_model = Attn_Unet(channels)

    model_path = os.listdir(args.snapshot_dir+"cloud_mask_model/")
    for filename in model_path: 
        filepath = os.path.join(args.snapshot_dir+"cloud_mask_model/", filename)       
        saved_state_dict = torch.load(filepath)
        cloud_mask_model.load_state_dict(saved_state_dict)   
        print("cloud_mask_model load state dict", filename)    
    model_path = os.listdir(args.snapshot_dir+"cloud_removal_model/")
    for filename in model_path: 
        filepath = os.path.join(args.snapshot_dir+"cloud_removal_model/", filename)       
        saved_state_dict = torch.load(filepath)
        cloud_removal_model.load_state_dict(saved_state_dict)   
        print("cloud_removal_model load state dict", filename)    

    real_input_imgs = []
    input_imgs = [input_img_t0, input_img_t1, input_img_t2, input_img_t3, input_img_t4, input_img_t5, input_img_t6, input_img_t7, input_img_t8, input_img_t9]
    T = int(T)
    for i in range (T):
        real_input_imgs.append(input_imgs[i])
    
    data_loader = data.DataLoader(
                    GradioData(real_input_imgs),
                    batch_size=1, shuffle=False)
        
    if cuda:
        cloud_mask_model = cloud_mask_model.cuda()
        cloud_removal_model = cloud_removal_model.cuda()

    images, cloud_masks = generate_cloud_mask(cuda, cloud_mask_model, data_loader)         # images (num, 3, 128, 128), cloud_masks (num, 128, 128)
    
    
    # fill cloud pixels with max intensity 1
    cloud_masks_bool = cloud_masks.clone().to(torch.bool).repeat(3, 1, 1, 1).transpose(0,1)  # cloud_masks [num, 3, 128, 128]
    images = images.masked_fill(cloud_masks_bool, 1)
    
    # T = images.shape[0]

    # padding
    if T < 10:
        num_frames_to_pad = 10 - T
        for i in range(num_frames_to_pad):
            images = torch.cat((images, torch.zeros_like(images[0]).unsqueeze(0)), 0)
            cloud_masks = torch.cat((cloud_masks, torch.zeros_like(cloud_masks[0]).unsqueeze(0)), 0)
    
    images = images.unsqueeze(0)
    pred_images = cloud_removal(cuda, cloud_removal_model, images)
    pred_images = pred_images.detach().cpu().numpy()
    
    output_imgs = []
    
    for idx in range(pred_images.shape[0]):        
        
        for t in range(T):
                                
            r = pred_images[idx][t,2,:,:]
            g = pred_images[idx][t,1,:,:]
            b = pred_images[idx][t,0,:,:]
            rgb_image = (np.dstack((r,g,b))*255).astype('uint8')
            output_imgs.append(rgb_image)
        
        if T < 10:
            for i in range(num_frames_to_pad):
                empty_img = np.ones_like(rgb_image)*255
                output_imgs.append(empty_img)
    
    return output_imgs[0], output_imgs[1], output_imgs[2], output_imgs[3], output_imgs[4], output_imgs[5], output_imgs[6], output_imgs[7], output_imgs[8], output_imgs[9]




with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Clear Vision
    Here's a multi-temporal satellite image cloud removal system. Allows you to remove clouds appearing in a satellite image time series.
    This system is optimized for sequences comprising 5 to 10 images with spatial size 128*128. For optimal performance, it's recommended to provide sequences where no more than half of the images are covered with clouds. 
    """)
    max_T = 10
    T = gr.Number(value = 10, label= "T", info = "Length of the time sequences", show_label = True, interactive = True, minimum = 5, maximum = 10)
    
    input_boxes = []
    output_boxes = []

    for i in range(max_T):
        with gr.Row().style(equal_height=True):
            input_imgs = gr.Image(shape=(128, 128), height = 200, width = 200, label = "input image index " + str(i), show_label = True)
            gr.Examples(
                examples=[os.path.join("./examples/", "original_t"+str(i)+ ".png")],
                inputs=input_imgs
            )
            input_boxes.append(input_imgs)
            output_imgs = gr.Image(height = 200, width = 200, label = "output image index " + str(i), show_label = True)
            output_boxes.append(output_imgs)
    
    input_boxes.append(T)
    with gr.Row().style(equal_height=True):
        clear_btn = gr.ClearButton(value = "Clear", components = input_boxes+output_boxes)
        submit_btn = gr.Button("Submit")
    
    submit_btn.click(fn=remove_cloud, inputs = input_boxes, outputs = output_boxes, api_name = "Cloud removal")
    
demo.launch(share=True)

