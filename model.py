import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch._utils
import torch.nn as nn
import math
# from .resnet import Bottleneck
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=True, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class ClassBlock2(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, return_f=False):
        super(ClassBlock2, self).__init__()
        self.return_f = return_f
        add_block = []
        # if linear:
        #     add_block += [nn.Linear(input_dim, num_bottleneck)]
        # else:
        # num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


#############ResNet50########################
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_test(nn.Module):

    def __init__(self, model):
        super(ft_net_test, self).__init__()
        model_ft = model.model
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x

################**Resnet101***######################
class Resnet101(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(Resnet101, self).__init__()
        model_ft = models.resnet101(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class Resnet101_test(nn.Module):

    def __init__(self, model):
        super(Resnet101_test, self).__init__()
        model_ft = model.model
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x

##############DenseNet121######################
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ft_net_dense_test(nn.Module):

    def __init__(self, model):
        super(ft_net_dense_test, self).__init__()
        self.model = model.model
        # model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # model_ft.fc = nn.Sequential()
        # self.model = model_ft

    def forward(self, x):
        x = self.model.features(x)
        # x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x
#################twostream_Resnet50###########################################
class twostream_Resnet50(nn.Module):
    def __init__(self, class_num):
        super(twostream_Resnet50, self).__init__()
        # submodel1
        model_ft1 = models.resnet50(pretrained=True)
        model_ft1.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft1.fc = nn.Sequential()
        self.submodel1 = model_ft1
        self.classifier = ClassBlock(2048, class_num, droprate=0.5, relu=True, bnorm=True, num_bottleneck=256)

        # submodel2
        self.part = 6  # We cut the pool5 to 6 parts
        model_ft2 = models.resnet50(pretrained=True)
        model_ft2.avgpool=nn.Sequential()
        model_ft2.fc = nn.Sequential()
        # self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        model_ft2.layer4[0].downsample[0].stride = (1, 1)
        model_ft2.layer4[0].conv2.stride = (1, 1)
        self.submodel2 = model_ft2
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=True, bnorm=True, num_bottleneck=256))

        # self.model=[model_ft1,model_ft2]

    def forward(self, xt):

        # submodel1
        x = self.submodel1.conv1(xt)
        x = self.submodel1.bn1(x)
        x = self.submodel1.relu(x)
        x = self.submodel1.maxpool(x)
        x = self.submodel1.layer1(x)
        x = self.submodel1.layer2(x)
        x = self.submodel1.layer3(x)
        x = self.submodel1.layer4(x)
        x = self.submodel1.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        y1 = self.classifier(x)

        # submodel2
        x = self.submodel2.conv1(xt)
        x = self.submodel2.bn1(x)
        x = self.submodel2.relu(x)
        x = self.submodel2.maxpool(x)
        x = self.submodel2.layer1(x)
        x = self.submodel2.layer2(x)
        x = self.submodel2.layer3(x)
        x = self.submodel2.layer4(x)
        x2 = self.avgpool(x)
        # x2 = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x2[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        y2 = []
        y2.append(y1)
        for i in range(self.part):
            y2.append(predict[i])

        return y2


class twostream_Resnet50_test(nn.Module):
    def __init__(self, model):
        super(twostream_Resnet50_test, self).__init__()
        self.submodel1 = model.submodel1
        # model_ft1.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # submodel2
        self.part = 6  # We cut the pool5 to 6 parts
        self.submodel2 = model.submodel2
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        # model_ft2.layer4[0].downsample[0].stride = (1, 1)
        # model_ft2.layer4[0].conv2.stride = (1, 1)

    def forward(self, xt):
        # submodel1
        x = self.submodel1.conv1(xt)
        x = self.submodel1.bn1(x)
        x = self.submodel1.relu(x)
        x = self.submodel1.maxpool(x)
        x = self.submodel1.layer1(x)
        x = self.submodel1.layer2(x)
        x = self.submodel1.layer3(x)
        x = self.submodel1.layer4(x)
        x1 = self.submodel1.avgpool(x)
        y0 = x1.view(x1.size(0), x1.size(1))
        y1 = torch.unsqueeze(y0, 2)

        # submodel2
        x = self.submodel2.conv1(xt)
        x = self.submodel2.bn1(x)
        x = self.submodel2.relu(x)
        x = self.submodel2.maxpool(x)
        x = self.submodel2.layer1(x)
        x = self.submodel2.layer2(x)
        x = self.submodel2.layer3(x)
        x = self.submodel2.layer4(x)
        x2 = self.avgpool(x)
        y2 = x2.view(x2.size(0), x2.size(1), x2.size(2))
        L5 = y2[:, :, 5]
        y = torch.cat((y1, y2), 2)

        return L5


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()
        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y

#################VGG16#####################
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16(nn.Module):

    def __init__(self, class_num, features=make_layers(cfg['D']), droprate=0.0):
        super(VGG16, self).__init__()
        self.features = features
        model_ft = models.vgg16(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(512, class_num, droprate)

    def forward(self, x):
        x = self.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class VGG16_test(nn.Module):

    def __init__(self, model):
        super(VGG16_test, self).__init__()
        model_ft = model.model
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

###################Inception_v3###################################
# Define the inception_v3-based Model
class Inceptionv3(nn.Module):

    def __init__(self, class_num, droprate=0.5, aux_logits=True, transform_input=False):
        super(Inceptionv3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        model_ft = models.inception_v3(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        # if self.training and self.aux_logits:
        #     aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)
        # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class Inceptionv3_test(nn.Module):

    def __init__(self, model):
        super(Inceptionv3_test, self).__init__()
        # self.aux_logits = True
        self.transform_input = False
        self.model = model.model
        # avg pooling to global pooling
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        # if self.training and self.aux_logits:
        #     aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)
        # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        x = self.model.avgpool(x)
        # 1 x 1 x 2048
        x = x.view(x.size(0), x.size(1))

        return x

