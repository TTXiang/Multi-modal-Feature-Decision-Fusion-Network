https://github.com/TTXiang/Multi-modal-Feature-Decision-Fusion-Network'

## Multi-dimensional optical image cross-modality fusion addresses the challenge of detecting faint targets in complex scenarios

Repository contains Python implementation of several methods for  models: 

* ViT-Fuse - new fuse method which gives better results comparing to others 
* Decision-Net - new decision fuse method which fuse four modes images
* my_dataloader - dataloader processing which read 9 channels images(only use 8 channels in the paper)

## Requirements

Python 3.*, Numpy, Numba, opencv


# Video demo/Datasets
* We provide two captured video streams with a resolution of 1280*750 fps = 10 to run as demos. 
* The images contain bounding boxes for object detection, and the confidence level of the object detection is indicated in the top left corner. 
* More data is pending for further open-sourcing.
* Filename: demo1_rgb.mp4 & demo1_atten.mp4; demo2_rgb.mp4
* Results shown in rgb and attenuation Images


More examples will be coming soon.