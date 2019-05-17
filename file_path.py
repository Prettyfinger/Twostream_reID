# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from model import ft_net, ft_net_test, ft_net_dense, ft_net_dense_test, VGG16, VGG16_test, PCB, PCB_test, \
    twostream_Resnet50, twostream_Resnet50_test, Inceptionv3, Inceptionv3_test
from model import Resnet101, Resnet101_test
parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--test_dir', default='/home/mcii216/fmx/dataset_ReID/Market-1501/pytorch', type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/home/mcii216/fmx/dataset_ReID/DukeMTMC-reID/pytorch',type=str, help='./test_data')
parser.add_argument('--test_dir',default='/home/mcii216/fmx/dataset_ReID/cuhk03/detected/pytorch',type=str, help='./test_data')
parser.add_argument('--batchsize', default=20, type=int, help='batchsize')
opt = parser.parse_args()

test_dir = opt.test_dir

data_transforms = transforms.Compose([
    transforms.Resize((384, 192), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()



# ######################################################################
# # Extract feature
# # ----------------------
# #
# # Extract feature from  a trained model.
#
#
# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         # filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2] == '-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels
#
#
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
f1 = open('txt/CUHK_querypath.txt', 'w')
f1.write("ID")
f1.write('   ')  # 添加一个空格
f1.write("Lable")  # 将图片的路径，写入文件
f1.write('   ')  # 添加一个空格
f1.write("camera")  # 将图片的路径，写入文件
f1.write('      ')  # 添加一个空格
f1.write("filename")  # 将图片的路径，写入文件
f1.write('               ')  # 添加一个空格
f1.write("path")  # 车牌号[标签]写入文件
f1.write('\n')  # 转行字符写入文件    换行
i = 0
for path, v in query_path:
    filename = os.path.basename(path)
    label = filename[0:4]
    camera = filename.split('c')[1]
    camera_num = int(camera[0])
    if label[0:2] == '-1':
        label=-1
    else:
        label=int(label)
    # camera_id.append(int(camera[0]))
    f1.write(str(i))
    f1.write('      ')  # 添加一个空格
    f1.write(str(label))  # 将图片的路径，写入文件
    f1.write('      ')  # 添加一个空格
    f1.write(str(camera_num))  # 将图片的路径，写入文件
    f1.write('          ')  # 添加一个空格
    f1.write(filename)  # 将图片的路径，写入文件
    f1.write('      ')  # 添加一个空格
    f1.write(path)  # 车牌号[标签]写入文件
    f1.write('\n')  # 转行字符写入文件    换行
    i = i+1

f2 = open('txt/CUHK_gallerypath.txt', 'w')
f2.write("ID")
f2.write('  ')  # 添加一个空格
f2.write("Lable") # 将图片的路径，写入文件
f2.write('   ')  # 添加一个空格
f2.write("camera")  # 将图片的路径，写入文件
f2.write('         ')  # 添加一个空格
f2.write("filename")  # 将图片的路径，写入文件
f2.write('               ')  # 添加一个空格
f2.write("path")  # 车牌号[标签]写入文件
f2.write('\n')  # 转行字符写入文件    换行
j = 0
for path, v in gallery_path:
    filename = os.path.basename(path)
    label = filename[0:4]
    camera = filename.split('c')[1]
    camera_num = int(camera[0])
    if label[0:2] == '-1':
        label=-1
    else:
        label=int(label)
    f2.write(str(j))
    f2.write('     ')  # 添加一个空格
    f2.write(str(label))  # 将图片的路径，写入文件
    f2.write('     ')  # 添加一个空格
    f2.write(str(camera_num))  # 将图片的路径，写入文件
    f2.write('        ')  # 添加一个空格
    f2.write(filename)  # 将图片的路径，写入文件
    f2.write('     ')  # 添加一个空格
    f2.write(path)  # 车牌号[标签]写入文件
    f2.write('\n')  # 转行字符写入文件    换行
    j = j+1
#
#
#
#     #第i个文件的    路径+名字
#     # 符串 例如：'...../img/00000000_GO12ODQ_0.png'
#
#      str_filenames=os.path.splitext(str_path)[0]  #分割路径，返回路径名和文件扩展名的元组
#         # 路径名=....../img/00000000_GO12ODQ_0  文件扩展名=.png
#
#         str_file=str_path.split('/')#将 路径字符串  分割  分隔符为 '/’ 返回值为list
#         filename_str=str_file[-1] #车牌号    “00000000_GO12ODQ_0”
#         filename_str=filename_str.split('-')[1]#分割字符串，取车牌“GO12ODQ”
#         f.write(str_path)#将图片的路径，写入文件
#         f.write(' ')#添加一个空格
#         f.write(filename_str) #车牌号[标签]写入文件
#         f.write('\n')#转行字符写入文件    换行
#
#
# gallery_cam, gallery_label = get_id(gallery_path)
# query_cam, query_label = get_id(query_path)
