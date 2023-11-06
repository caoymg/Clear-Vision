# Clear Vision
 A end-to-end system of cloud removal for satellite images.




**Clear Vision Documentation**
Yiming CAO





**1. System Usage Guidelines**



**1.1 Environment requirements**
The system operates efficiently on the Microsoft Azure Linux (ubuntu 20.04) Virtual Machine with one GPU Tesla T4 (16384 MB). To ensure the optimal functioning of the system, the following software versions are recommended:
earthnet==0.3.9gradio==3.41.2gradio-client==0.5.0matplotlib==3.7.2numpy==1.24.4pandas==2.0.3torch==2.0.1torchvision==0.15.2Python version 3.8.10

To start the system you can simply run “run_gradio.py” in codes/Clear_Vision.

**1.2 Input Requirements**
For seamless operation of the AI-based system, adhere to the following input specifications:Spatial Satellite Image: RGB Images with 128 x 128 pixels. The image utilized for training and testing possesses dimensions of 128x128 pixels with a resolution of 20 meters. It is strongly recommended that the input images maintain this resolution (Inadequate input image quality, such as blurring after resizing one small resolution image, may render the cloud mask categorizing regular scenes as entirely cloud-covered, impacting model performance).Sequence Length: The length of the input image sequence should fall within the range of 5 to 10 images.Cloud Coverage: To attain optimal system performance, refrain from having clouds present in more than 50% of the images within the provided time sequence.Time Interval: There is no strict constraint on the time interval between images. However, it is imperative that images are arranged in the correct order. For enhanced system efficiency, it is recommended to maintain a relatively smaller time interval between consecutive images.



**1.3 System Performance Evaluation**
The performance metrics of the system are as follows:
PSNR (Peak Signal-to-Noise Ratio): The system achieves a PSNR of 34.894.RMSE (Root Mean Square Error): The RMSE achieved by the system is 0.021.
These results are based on an evaluation conducted using simulated data generated from the Earthnet2021 iid test dataset. The generation process adheres rigorously to the guidelines outlined in the simulation of data gaps as described in [1].
Furthermore, the system's efficacy is demonstrated through its performance on time series extracted from the EarthNet2021 ood test dataset. Notably, these time series contain real clouds, and the model has had no prior exposure to this data during training or testing.
Examples from EarthNet2021 ood Test Dataset with Real Clouds
Example 1:Input cloudy images

![img](https://lh7-us.googleusercontent.com/EvH0vvuuUzWznTe1YRGwm42NmLplexKx6FR-LYC7-8mVrTE0rRjvUgSp9HWw9fIUHk9ESaXliwSo_DgRjLlZBqSDgTONw_b9xFoKTMDONLsTKfEcJp0UENiaOVNjXnfVJksGadkNIkoH286kx422i9COMu5k-qo_FItF2UAioy70RkLgRhpJHCWwQYIAPg)![img](https://lh7-us.googleusercontent.com/49CsGiwnSKT8FJHGLx95dzklOGB6-LIp4DzXCA4bIBWfn_IsURDhE-UohvccsNzhldrTb7NgNY1Ri8WGKfISOtgqMWDhXEB-_4Pxcimqeq939L5mRECS5LuDJ80FqHkQd6mXYKkpr70Y3i3dmVtzYxMR1y3o4J11XDyDGB4A8IeGZvZBv4M6IzdNQCfbBA)![img](https://lh7-us.googleusercontent.com/Dm-E7gSqCX-hA2Udp8hgG6hLFdwGsG00XELQUbT-7WrdTdDVdkNmgdFA6jybQoB8A54ap7f-Uaf3q50m4SdKZSoRXIoUgNDDhSQQil4vWw6mahch7sk9HZRayDYssRe73Gr_QuWzAIkBrqCgVCR6lKs2k9bmD7KzYAptz7aXKeb4V_rUvhZMNodgXwD8gA)![img](https://lh7-us.googleusercontent.com/vvYOImRXDswqubpIQF6E1xxoXZsssN6pIgcuyrrqG4qXEmK3UhB939VxmqrWFCj7EJXQWOuTJgI2_6XTfPk_CR0zvJzE3g7c06l_EuT4Hxp2lcWJ_-8kdN0jTC0NfKX4gD0WYelXwmw1PVXuzC2vlUSMp2yeSTti-QkMMItPKKcRTwCKm98ZMxkiKIlIPQ)![img](https://lh7-us.googleusercontent.com/Wf0P9aWzU-zJ9-Rtt-y-W8ttkb3kYZUhL0jp6EznWiV7PT0NOYOAQBUbnJWZWzAdy1lEAz2q59wwQZ1UGEncBGwalHHsIbNmQRLkMz5mJZKwtCz6RdW7YpY1zxw4YfQySH8UgwZj1cJ_snGHGJOtQBuLo2aGova67qFXM-ZjsQ1QIm57z-uKDXWEISY-WQ)

Cloud-free images generated by Clear Vision

![img](https://lh7-us.googleusercontent.com/gvCG9qk6-xkZLRdmIQW7yCrDrH5_SZUYpt8eHtFFkaS0kwppOYvxUCgXqbvN60Woik-TnFkaXrXFAaUS9S3UgLWnfy7L6VcXyko7sdP_mPmhojg4f4bgY9M-k2Lf6_x0K6Vmg94Ww_il7Hm_kYVY62iWIgXLwROrXo12gpyxeOO36-0_mZBLZUL7oY2RTw)![img](https://lh7-us.googleusercontent.com/z8vYpg3HDVsAqKxPvD9H027M342HHLw2zld8us8OZ-VsuK99j8OjBxKgdrJe3w7Dlizm400awBRJqLIiZmf2phWDY4BBpTlwqn1KQKv24bFIyC5KgbrBCDp51czi798RW6DE1IYjih2TQhXPXFFxzVkqTl3-Oo54i83MoAVocly6S56stZ9RmnYFd5cVcA)![img](https://lh7-us.googleusercontent.com/uP8_SxRdVv4QsliZRj8Ss4VDfENvZ9SBBt1__usCtU02t_4cRGucUcuHwgcKxoPc70jxq-FaC9ylFC_rrf4lLBDnkHt3qFusTTzO_9aRR_vkzOL-gcwtFdj_Qz939GlpqcCBgi90OB_2PxmmRWTDqxE6rNkADDQf27YjCe9fX7qMCv6KcaMnhN-6NidajA)![img](https://lh7-us.googleusercontent.com/t5FtHmEU1HIe-anohug8Ih4yTM21zoESb4CEFKjULOsrlXgkXL4bOfa6wqzI9O9TGqUz-kmNZZVa4NEcreIQ62xGpdPnhi84Q49JX_KaQOWJ811Bi4wWmMON1k8w8m5vszHB1W8m3ZzQBTD3A-KmkQI1lCkSX-16YnBdvVcpyhb7TbTjBN6ZgKfcw0_2VQ)![img](https://lh7-us.googleusercontent.com/vAJT4EqOU6Wm1jz1HDCWjJqgLpT5zRNDn6JoSaIrPNXUMvnq8utLFVGQJNeq7CbA-LSC6Zct1zezigDmg2Y_taZI5th0QZcimi6Pn1nYSkzqZBHRtWyfDv0l7PLxpGEXwFwZm7v_cyH5Y2HYVJHbdvw7ZYyjbCNoEByVSIC89pRQ38u_UzucoO5wDbSHiA)

Example 2:Input cloudy images

![img](https://lh7-us.googleusercontent.com/0JCfraCCtN62K-zC_YCAmBwzpyOc2sjjUWv_nXYW7--YiT3qOM5taSwQHrJQ1blzp2rS11e4JWsB1KySYtDA8oqHUZV-XHkijQEsOOEWsfCVQYXQOU4nieR0nWjU13PKhGkcFUK-rtbdOh4iITI-nR7SE1QNz63iw8aenTD27w4N-lJIIHMq6gpQphBPoA)![img](https://lh7-us.googleusercontent.com/sPx8lIbx6amsf-tSlC7lWao2zz3CrvMmElQVXEE8YMEpwqJPcFdJIRxUZNnyMLrA1o8rUgR_tEH1kt4UxtqxTwv8EZmvrXjiDL8AzShIK0n_1qDnaGqGJhrIheByoXiQFloTk1Q2MT_osRD8p3B3-p5ZOXRwxFsFfep35gUm0YgwuMbow5FiBSdNseKf0g)![img](https://lh7-us.googleusercontent.com/xwZCUn6-7XLFK2eh-tD8z0oqw2vn2XWwzoQnsFal7ghgsXoXLY926qNd-A0QF-58pAUi6H5Tb0NKUh6Bx27T3MkeWWJ8McirsGRdvSupvWdocq1nJL9etZcIPOT0I6tXXTDolpUqy3peSst4riZOzlOfD5t_-NINc9lKwwm9YBJN36dJgMX5tzqU1uFwvQ)![img](https://lh7-us.googleusercontent.com/byF4Fz3C_JXgGI3m0I9oupDH8PeUu4ov50YFHCjCq1zz3TICnYXq0QpznUpY4-vMkDfyZnTqctJ2-Tp1sHbKo_keRQ0Qi7x1NxtTECmEB9w8zOGhWOzSiBDiEzIlZ69iuq2Ay7vyLl8qMJLGV6hJKmThdfBHMZiheqSaOWNQ4EMWRESkayZ1UFfTVaC9Xw)![img](https://lh7-us.googleusercontent.com/JSn54IF0lh3IoKEAF3aHeMday-b_5q2xaXTCSLjhuIog2c2CMXqDGABD4r_xjZAhftFtce11Z4tLv7_HBANX8W1Qe_g5F6N0ksi6DZPem9wckwMKQ1ZQ14pek8bjUNSUrQrFKfEX1SQo1sDKRPAJ7lmPuQ7ILfXOUQrNKtXOhNMOBR67RW2ItGk31PftgA)

Cloud-free images generated by Clear Vision

![img](https://lh7-us.googleusercontent.com/tSeCMuIE3QAVAJxnN2PqNWFx4kw_VfoXTD5aHFXVbhyYAIMaHIpt7kfTTGP2q_Podek7l5JIbartNLF2l7iECLa6e2ArEhhyQw_GaxUvpNNMGO73-N6EapfnfYmYSkU3aBzmS77QMsPPg_z6-XSlRMRiOKU_6CUHqbUt3tNZHNqBct3ez_v22Xx-NbjVaQ)![img](https://lh7-us.googleusercontent.com/f2vOzCfc-1kZdZ6LAllGCgO-WJHA31FrF8ndluw0vY_2cR50NQbc4GDJLaxJCPBUfxar7x86BFEUFyTI7Y966Pp6FQcTk17IcSxRmToxsfdBLwEEiaiyyEQn0HPtzT_MLhSm5_OTu-9HS8KgpkgvBjoGcnpPSsoHkZfKw59vMi-O69dOFcRPYSNL-VeAvA)![img](https://lh7-us.googleusercontent.com/nd0_BFI1lilY3cjFur0trPcecatPjKlgS1L4ATv8ndC6n_APYxXywEz4h-IVSG8Oqkwu9KwcXOU6YLYXPG_prvbmlNeOmOPIHCm4CLClD6_GtkkPgcVEkQvAcKtyptONENkWMNybUMmJTHkA7xczeDqHIE4NfEbPSkI1HFuY01lcaeBHg5KaaEiHQaUh4A)![img](https://lh7-us.googleusercontent.com/AKNw1nrNMPZMOvrBo9kI_nUUO9Z3v-HTJnbVCpeoPa-AAnrHyp5j6URx4-Srk_juMUn5JqxAT0dd1XT-Ebt5KkTrp4cjFZbcdqZffz0jYwLIIj2YVPlKkib2mjZfqmjBpdQ0Qjv2FFXnxMlv4kIUeo5VEDHIlGwvdDF2_tE2s_cmG_3CxHcMfdNq7RTzHQ)![img](https://lh7-us.googleusercontent.com/rnTAmhsHr26lmVvuCfTy-VmfM0rtSBOJyTxC9elCI7zWenyrCtj6XCJaAH92cH97kb95pIz5ysOb0dtmdhKKRwPiW7Bl68avfogWOms172LEpqWa8UxYAULOm4Y1V1h0GvSDulGjyYriEzOa0dUKJPipIrsxiyvQEL5cS0q-YsxGD9YB-18-XAtG-TpKoQ)



**1.4 Current Limitations**
While the cloud detection method employed by the system is effective in many scenarios, it is important to note a specific limitation. 
First, the method may overlook the presence of clouds when the clouds are thin, resulting in the ensuing cloud removal procedure incapable of addressing their elimination. 
Second, the system does not incorporate exact time information into its processes. Consequently, in scenarios where consecutive images are fully cloaked in clouds and fed into the system, the system produces identical predictions for both instances. This is contrary to the anticipated outcome of gradually changing predictions that account for temporal evolution.

**2. System Architecture**
The system is constructed as a two-stage cohesive model integrated within the Gradio framework. The two stages are cloud detection and cloud removal executed by separate models.
Stage 1: Cloud DetectionIn this stage, the input image undergoes processing through a U-Net-based semantic segmentation model [2]. The outcome is the generation of a cloud mask that identifies cloud-covered areas.
Stage 2: Cloud RemovalThe subsequent stage involves two inputs: the generated cloud mask and the original input image. These inputs are directed into a symmetric multi-scale encoding and decoding model. The cloud removal model integrates temporal attention and attention-weighted skip connections, which collectively contribute to the removal of clouds. The cloud removal model is based on the architecture proposed in [1], enhanced with an augmented model capacity in both downsampling and upsampling channel depth. 


**3. File Structure**
“Clear_Vision” Folder:Contains all the code for system execution."examples" folder: This directory houses illustrative cloudy images showcased within the Gradio interface."networks" folder: Within this repository resides the model code for cloud detection and cloud removal."model_dict" folder: This repository contains the pre-trained model parameters for cloud detection and cloud removal functionalities.
“Preprocess” Folder:Contained within the "Preprocess" folder is the preprocessing pipeline for the Earthnet2021 dataset. This entails tasks such as RGB channel extraction and cloud mask generation. For both training and evaluation datasets, critical steps involve selecting valid time series and generating simulated data by merging meticulously selected cloud-free frames with authentic cloud masks.
“Cloud_mask_pretrain” Folder:The "Cloud_mask_pretrain" folder encompasses the training code for the cloud mask generation model. Should you desire to tailor the model structure, modifications can be made in "network.py". Additionally, the "dataloader.py" file can be adjusted to accommodate custom data for training or evaluation. To initiate model training and evaluation, execute "run.py". 
“Cloud_removal_pretrain” Folder:Similarly, the "Cloud_removal_pretrain" folder houses the training code for the cloud removal model. "network.py" can be edited to adjust the model's architecture, and "dataloader.py" can be modified to incorporate specific training or evaluation data. Model training and evaluation are executed using "run.py".







**Reference**

[1] Stucker C, Sainte Fare Garnot V, Schindler K. U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series[J]. arXiv e-prints, 2023: arXiv: 2305.13277.[2] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015: 234-241.