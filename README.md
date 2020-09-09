# EDVR分支
在EDVR分支下，将PFNL、DUF等tensorflow代码转为了torch的，并添加了GAN loss（WGAN、RGAN、LSGAN）、MS-SSIMloss、edge loss（ours）、clip L1 loss等

# EDVR_YK分支
通过优酷的数据训练了全新的模型，由于其数据中包含转场，不利于使用多帧重建出一帧，需要现记录下训练数据中的转场位置。转场算法在Trans分支下

# trans分支
只需要给定视频就可以一键生成对应的转场文件，在训练过程中识别转换场景的位置。
