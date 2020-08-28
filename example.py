#!/usr/bin/env python
# coding: utf-8

# # TransNet: A deep network for fast detection of common shot transitions
# This repository contains code for paper *TransNet: A deep network for fast detection of common shot transitions*.
# 
# If you use it in your work, please cite:
# 
# 
#     @article{soucek2019transnet,
#         title={TransNet: A deep network for fast detection of common shot transitions},
#         author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Moravec, Jaroslav and Loko{\v{c}}, Jakub},
#         journal={arXiv preprint arXiv:1906.03363},
#         year={2019}
#     }

# ## How to use it?
# 
# Firstly, *tensorflow* needs to be installed.
# Do so by doing:
# 
#     pip install tensorflow
# 
# If you want to run **TransNet** directly on video files, *ffmpeg* needs to be installed as well:
# 
#     pip install ffmpeg-python
# 
# You can also install *pillow* for visualization:
# 
#     pip install pillow
# 
#     
# Tested with *tensorflow* v1.12.0.

# In[1]:


import ffmpeg
import numpy as np
import tensorflow as tf

from transnet import TransNetParams, TransNet
from transnet_utils import draw_video_with_predictions, scenes_from_predictions
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# In[2]:


# initialize the network
params = TransNetParams()
params.CHECKPOINT_PATH = "./model/transnet_model-F16_L3_S2_D256"

net = TransNet(params)


# In[3]:
paths = '/data/zhoubo/videodataset/Youku/LR/'
files = os.listdir(paths)

# export video into numpy array using ffmpeg
f = open('result.txt','w+')
for file in files:
# if True:
    # src = 'testvideo/'+os.path.splitext(file)[0]+'.mp4'
    filename = os.path.splitext(file)[0]
    src = paths+file
    print(src)
    
    video_stream, err = (
        ffmpeg
        .input(src)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
        .run(capture_stdout=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])


    # In[4]:


    # predict transitions using the neural network
    predictions = net.predict_video(video)


    # ## Visualize results
    # 
    # Function `draw_video_with_predictions` displays all video frames with confidence bars for each frame. The green bar is considered as detected shot boundary (predicted value exceeds the threshold), the red bar is shown otherwise.
    # 
    # Function `scenes_from_predictions` returns a list of scenes for a given video. Each scene is defined as a tuple of (start frame, end frame).
    # 
    # As described in the paper, the threshold of `0.1` is used.

    # In[5]:


    # For ilustration purposes, we show only 200 frames starting with frame number 8000.
    img = draw_video_with_predictions(video, predictions, threshold=0.1)
    
    # filename = os.path.splitext(file)[0]
    img.save('result/'+filename+'.png', quality=95)
    # In[6]:


    # Generate list of scenes from predictions, returns tuples of (start frame, end frame)
    scenes = scenes_from_predictions(predictions, threshold=0.1)

    # For ilustration purposes, only the visualized scenes are shown.
    f.write(filename)
    f.write(':\n')
    f.write(str(scenes))
    f.write('\n')
    print(scenes)
f.close()

