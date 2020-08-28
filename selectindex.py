import numpy as np
import random

transindex = np.load('/data/zhoubo/TransNet-master/youku.npy',allow_pickle=True)

num = random.randint(0,999)

frames = random.randint(0,99)

n_frames = 5

n_half = 2

# flag = True
# for te in range(1000000):
#     num = random.randint(0,999)
#     each_trans = transindex[num]
#     # print(num)
#     flag = True
#     while(flag):
#         for i in each_trans:
#             if frames+n_half<i[1] and frames-n_half>i[0]:
#                 # print('一切正确',num,each_trans,frames)
#                 flag = False
#                 break
#         if flag:
#             frames = random.randint(0,99)
#     testflag = False
#     for i in each_trans:
#         if frames-n_half>i[0] and frames+n_half<i[1]:
#             testflag = True
#     if not testflag:
#         print('出现错误')
#         print('video:',num,each_trans,frames)

def frame_num(each_trans,frames,n_half):
    flag = True
    while(flag):
        for i in each_trans:
            if frames+n_half<i[1] and frames-n_half>i[0]:
                # print('一切正确',num,each_trans,frames)
                flag = False
                break
        if flag:
            frames = random.randint(0,99)
    return frames
for i in range(1000):
    num = random.randint(0,999)
    frames = random.randint(0,99)

    each_trans = transindex[num]
    center_frame_idx = frame_num(each_trans,frames,n_half)

    for i in each_trans:
        if center_frame_idx-n_half>i[0] and center_frame_idx+n_half<i[1]:
            testflag = True
    if not testflag:
        print('出现错误')
        print('video:',num,each_trans,center_frame_idx)