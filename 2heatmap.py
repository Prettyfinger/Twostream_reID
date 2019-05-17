import cv2
import os
import numpy as np
import cv2 as cv
import numpy as np
from torchvision import models
img = cv2.imread('picture/Dq0005.jpg', 1)  # 原图
# img = cv2.imread('R18_picture/M1.jpg', 1)
# hc=[[[]]]
# hg=[[[]]]
for i in range(20):
    att = cv2.imread(os.path.join('picture/submodel2.layer4', str(i+1)+'.png'), 1)
    # att = cv2.imread(os.path.join('R18_picture/model.avgpool', str(i+1)+'.png'), 1)
    att = cv2.resize(att, (img.shape[1], img.shape[0]))
    # cv2.imwrite(os.path.join('picture', 'att.jpg'), att)
    w = cv2.applyColorMap(att, 2)  # 转化为jet 的colormap
    # cv2.imwrite(os.path.join('picture', 'w.jpg'), w)
    x = img * 0.4 + w * 0.6  # 权重自己定
    xc = x.astype(np.uint8)             #colormap
    ##########color to gray##########
    gray=np.array(xc)
    gray=gray[:,:,0]
    new_color=np.array([gray,gray,gray])
    xg=np.transpose(new_color,(1,2,0))   #graymap

    # cv2.imwrite(os.path.join('R18_picture/model.avgpool.feacolor', str(i+1)+'.jpg'), x)
    cv2.imwrite(os.path.join('picture/submodel1.layer1.feacolor', str(i+1)+'.jpg'), xc)
    cv2.imwrite(os.path.join('picture/submodel1.layer1.feagray', str(i+1)+'.jpg'), xg)
    # att+=att
    # xc+=xc
    # xg+=xg
# cv2.imwrite(os.path.join('picture/heatmap/submodel2.layer4f.jpg'), att)
# cv2.imwrite(os.path.join('picture/heatmap/submodel2.layer4c.jpg'), xc)
# cv2.imwrite(os.path.join('picture/heatmap/submodel2.layer4g.jpg'), xg)
# #










# for filename in os.listdir(r"picture/submodel2.avgpool/"):
#     att=cv2.imread(os.path.join('picture/submodel2.avgpool/', filename), 1)
#     att = cv2.resize(att, (img.shape[1], img.shape[0]))
#     w = cv2.applyColorMap(att, 2)  # 转化为jet 的colormap
#     x = img * 0.4 + w * 0.6  # 权重自己定
#     x = x.astype(np.uint8)
    # gray=np.array(x)
    # gray=gray[:,:,0]
    # new_color=np.array([gray,gray,gray])
    # x=np.transpose(new_color,(1,2,0))
    # i=i+1
    # cv2.imwrite(os.path.join('picture/submodel2.avgpool.fea/', str(i), '.jpg'), x)


# att = cv2.imread('picture/submodel2.avgpool/2.png', 1)  # feature map
# print(att.shape)
# att= cv2.resize(att, (img.shape[1], img.shape[0]))
# print(att.shape)
# w = cv2.applyColorMap(att, 2)  # 转化为jet 的colormap
# print(w.shape)
# # a=cv2.imread('demo_att.jpg', 1)
# # print(a.shape)
# x = img * 0.4 + w * 0.6  # 权重自己定
# x = x.astype(np.uint8)
# # gray=np.array(x)
# # gray=gray[:,:,0]
# # new_color=np.array([gray,gray,gray])
# # x=np.transpose(new_color,(1,2,0))
# cv2.imwrite('picture/f1.jpg', x)
# # print('###########')
# # img1 = cv2.imread('picture/submodel2.avgpool/20.png')  # 原图
# # print(img1)
# # print('###########')
# # img2 = cv2.imread('picture/submodel2.layer4/1.png')  # 原图
# # x = img2.astype(np.uint8)
# # gray=np.array(x)
# # gray=gray[:,:,0]
# # new_color=np.array([gray,gray,gray])
# # x=np.transpose(new_color,(1,2,0))
# # cv2.imwrite('picture/f2.jpg', x)


