
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.nn import init
import os
import scipy.io
import yaml
from model import ft_net, ft_net_test, ft_net_dense, ft_net_dense_test,  VGG16, VGG16_test, PCB, PCB_test, twostream_Resnet50, twostream_Resnet50_test,Inception_v3, Inception_v3_test, Resnet50_18, Resnet50_18_test, twostream_Resnet50_18, twostream_Resnet50_18_test, Resnet50_12, Resnet50_12_test,twostream_Resnet50_12,twostream_Resnet50_12_test
# from model import tmpncovresnet50, tmpncovresnet50_test, twostream_PCBCOV, twostream_PCBCOV_test, Resnet101, Resnet101_test
# from model import twostream_Resnet101, twostream_Resnet101_test, Inception_v3, Inception_v3_test
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and testing datasets.
pic_dir = 'picture/Dq0005.jpg'
# pic_dir = 'R18_picture/M1.jpg'

# Market1501: 751
# DUKE : 702
# CUHK03D: 767
class_num=702
# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()

# --------load training model-------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='twostream_DUKE', type=str, help='save model path')
parser.add_argument('--twostream_Resnet50', action='store_true', help='use twostream+ResNet50')
parser.add_argument('--Resnet50_18', action='store_true', help='use ResNet50_18')

opt = parser.parse_args()

###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.twostream_resnet50 = config['twostream_Resnet50']
opt.Resnet50_18 = config['Resnet50_18']

name = opt.name

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (384, 192))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)
#
#
# def get_picture_rgb(picture_dir):
#     '''
#     该函数实现了显示图片的RGB三通道颜色
#     '''
#     img = skimage.io.imread(picture_dir)
#     img256 = skimage.transform.resize(img, (384, 192))
#     skimage.io.imsave('./picture/orimage.jpg', img256)
#
#    # 取单一通道值显示
#     for i in range(3):
#         img = img256[:,:,i]
#         ax = plt.subplot(1, 3, i + 1)
#         ax.set_title('Feature {}'.format(i))
#         ax.axis('off')
#         plt.imshow(img)
#
#     r = img256.copy()
#     r[:, :, 1:3] = 0
#     ax = plt.subplot(1, 4, 3)
#     ax.set_title('R Channel')
#     # ax.axis('off')
#     plt.imshow(r)
#     plt.imsave('picture/R.jpg', r)
#
#     g = img256.copy()
#     g[:,:,0]=0
#     g[:,:,2]=0
#     ax = plt.subplot(1, 4, 2)
#     ax.set_title('G Channel')
#     # ax.axis('off')
#     plt.imshow(g)
#     plt.imsave('picture/G.jpg', g)
#
#     b = img256.copy()
#     b[:, :, 0:2] = 0
#     ax = plt.subplot(1, 4, 1)
#     ax.set_title('B Channel')
#     # ax.axis('off')
#     plt.imshow(b)
#     plt.imsave('picture/B.jpg', b)
#
#     img = img256.copy()
#     ax = plt.subplot(1, 4, 4)
#     ax.set_title('image')
#     # ax.axis('off')
#     plt.imshow(img)
#
#     img = img256.copy()
#     ax = plt.subplot()
#     ax.set_title('image')
#     # ax.axis('off')
#     plt.imshow(img)
#
#     plt.show()


# class LeNet(nn.Module):
#     '''
#     该类继承了torch.nn.Modul类
#     构建LeNet神经网络模型
#     '''
#
#     def __init__(self):
#         super(LeNet, self).__init__()
#
#         # 第一层神经网络，包括卷积层、线性激活函数、池化层
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 1, 2),  # input_size=(3*256*256)，padding=2
#             nn.ReLU(),  # input_size=(32*256*256)
#             nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(32*128*128)
#         )
#
#         # 第二层神经网络，包括卷积层、线性激活函数、池化层
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, 1, 2),  # input_size=(32*128*128)
#             nn.ReLU(),  # input_size=(64*128*128)
#             nn.MaxPool2d(2, 2)  # output_size=(64*64*64)
#         )
#
#         # 全连接层(将神经网络的神经元的多维输出转化为一维)
#         self.fc1 = nn.Sequential(
#             nn.Linear(64 * 64 * 64, 128),  # 进行线性变换
#             nn.ReLU()  # 进行ReLu激活
#         )
#
#         # 输出层(将全连接层的一维输出进行处理)
#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 84),
#             nn.ReLU()
#         )
#
#         # 将输出层的数据进行分类(输出预测值)
#         self.fc3 = nn.Linear(84, 62)
#
#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x


# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, module, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.module = module
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        # x=self.extracted_layers(x)
        # print(self.submodule._modules.items())
        # for name, module in self.submodule._modules.items():
            # if "fc" in name:
            #     print(name)
            #     x = x.view(x.size(0), -1)
            # print(module)
            # x = module(x)
            # print(name)
            # if name in self.extracted_layers:
        # submodel1
        x = self.module.submodel1.conv1(x)
        x = self.module.submodel1.bn1(x)
        x = self.module.submodel1.relu(x)
        x = self.module.submodel1.maxpool(x)
        x = self.module.submodel1.layer1(x)
        x = self.module.submodel1.layer2(x)
        x = self.module.submodel1.layer3(x)
        x = self.module.submodel1.layer4(x)
        x = self.module.submodel1.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        # y1 = self.classifier(x)
        #
        # # submodel2
        # x = self.module.submodel2.conv1(x)
        # x = self.module.submodel2.bn1(x)
        # x = self.module.submodel2.relu(x)
        # x = self.module.submodel2.maxpool(x)
        # x = self.module.submodel2.layer1(x)
        # x = self.module.submodel2.layer2(x)
        # x = self.module.submodel2.layer3(x)
        # x = self.module.submodel2.layer4(x)
        # x = self.module.avgpool(x)
        # x2 = self.dropout(x)
        # x=self.module.submodel1.conv1(x)
        # x = self.module.submodel2.avgpool(x)

        #Resnet_18 model
        # x = self.module.model.conv1(x)
        # x = self.module.model.bn1(x)
        # x = self.module.model.relu(x)
        # x = self.module.model.maxpool(x)
        # x = self.module.model.layer1(x)
        # x = self.module.model.layer2(x)
        # x = self.module.model.layer3(x)
        # x = self.module.model.layer4(x)
        # x = self.module.avgpool(x)
        outputs.append(x)
        # outputs.append(x)
        return outputs

# Load model#---------------------------
def load_network(network):
    save_path = os.path.join('./model', name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

#**********generate feature maps**********
def get_feature():
    # 输入数据
    img = get_picture(pic_dir, transform)
    # 插入维度
    img = img.unsqueeze(0)

    img = img.to(device)

    # 特征输出
    if opt.twostream_Resnet50:
        model_structure = twostream_Resnet50(class_num).to(device)
        model = load_network(model_structure)
        model = model.eval()
        exact_list = ["submodel1.avgpool"]
        myexactor = FeatureExtractor(model, exact_list)
        x = myexactor(img)

    else:
        model_structure = Resnet50_18(class_num).to(device)
        model = load_network(model_structure)
        model = model.eval()
        exact_list = ["submodel1.layer4"]
        myexactor = FeatureExtractor(model, exact_list)
        x = myexactor(img)

    for i in range(2048):
        ax = plt.subplot(50, 50,  i+1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        plt.imshow(x[0].data.cpu().numpy()[0, i, :, :], cmap='jet')
        y=x[0].data.cpu().numpy()[0, i, :, :]
        j= str(i+1)
        plt.imsave(os.path.join('picture/submodel1.avgpool', j), y)
    plt.show()


# 训练
if __name__ == "__main__":
    # get_picture_rgb(pic_dir)
    get_feature()

