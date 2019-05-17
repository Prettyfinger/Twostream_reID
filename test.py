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
from model import ft_net, ft_net_test, ft_net_dense, ft_net_dense_test, Resnet101, Resnet101_test,  VGG16, VGG16_test, PCB, PCB_test, twostream_Resnet50, twostream_Resnet50_test,Inceptionv3, Inceptionv3_test
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/mcii216/fmx/dataset_ReID/Market-1501/pytorch',type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/home/mcii216/fmx/dataset_ReID/DukeMTMC-reID/pytorch',type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/home/mcii216/fmx/dataset_ReID/cuhk03/detected/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='SEResnet50_Market1501', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--ft_net', action='store_true', help='use ft_net')
parser.add_argument('--ft_net_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--twostream_Resnet50', action='store_true', help='use twostream+ResNet50')
parser.add_argument('--Inceptionv3', action='store_true', help='use Inceptionv3' )
parser.add_argument('--VGG16', action='store_true', help='use VGG16')
parser.add_argument('--Resnet101', action='store_true', help='use Resnet101' )
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.ft_net = config['ft_net']
opt.ft_net_dense = config['ft_net_dense']
opt.PCB = config['PCB']
opt.twostream_Resnet50 = config['twostream_Resnet50']
opt.Inceptionv3 = config['Inceptionv3']
opt.VGG16 = config['VGG16']
opt.Resnet101 = config['Resnet101']
opt.ft_net_denseft_net_dense = config['ft_net_dense']
str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model#
# ---------------------------
# def load_network(network):
#     if opt.resume:
#         if os.path.isfile(os.path.join('./model',name, opt.resume)):
#             checkpoint = torch.load(os.path.join('./model',name, opt.resume))
#             opt.start_epoch = checkpoint['epoch']
#             best_acc = checkpoint['best_prec1']
#             network.load_state_dict(checkpoint['state_dict'])
#             return network

def load_network(network):
    save_path = os.path.join('./model', name, 'net_99.pth')
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        if opt.ft_net :  #Resnet50
            ff = torch.FloatTensor(n, 2048).zero_()
        elif opt.ft_net_dense:  #Densenet121
            ff = torch.FloatTensor(n, 1024).zero_()
        elif opt.VGG16:
            ff = torch.FloatTensor(n, 512).zero_()
        elif opt.Resnet101:
            ff = torch.FloatTensor(n, 2048).zero_()
        elif opt.twostream_Resnet50:
            ff = torch.FloatTensor(n, 2048).zero_()
        elif opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_() # we have six parts
        else:
            # ff = torch.FloatTensor(n, 1024).zero_()
            print('error')
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu().float()
            ff = ff+f
        # norm feature
        if opt.ft_net or opt.ft_net_dense or opt.Resnet101:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        elif opt.twostream_Resnet50:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        elif opt.PCB:
            # feature size (n,2048,6)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)
if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)
######################################################################
# Load Collected data Trained model
# Market1501: 751
# DUKE : 702
# CUHK03D: 767
print('-------test-----------')
if opt.ft_net:
    model_structure = ft_net(767)
elif opt.ft_net_dense:
    model_structure = ft_net_dense(767)
elif opt.PCB:
    model_structure = PCB(702)
elif opt.twostream_Resnet50:
    model_structure = twostream_Resnet50(767)
elif opt.VGG16:
    model_structure= VGG16(751)
elif opt.Inception_v3:
    model_structure=Inceptionv3(767)
elif opt.Resnet101:
    model_structure = Resnet101(767)
else:
    print("There is no model!")

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    model = PCB_test(model)
    # model = network_to_half(model)
elif opt.twostream_Resnet50:
    model = twostream_Resnet50_test(model)
elif opt.VGG16:
    model=VGG16_test(model)
elif opt.Inceptionv3:
    model=Inceptionv3_test(model)
elif opt.ft_net:
    model=ft_net_test(model)
elif opt.ft_net_dense:
    model=ft_net_dense_test(model)
elif opt.Resnet101:
    model = Resnet101_test(model)
else:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])

result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('mat/VGG16_Market1501.mat',result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)
