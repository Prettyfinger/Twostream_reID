# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch.utils.model_zoo as model_zoo
import argparse
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import warnings
from model import ft_net, ft_net_dense,  PCB, VGG16, twostream_Resnet50, Resnet101, Inceptionv3
from random_erasing import RandomErasing
import yaml
import shutil
from shutil import copyfile

model_urls = {
    'mpncovresnet50': 'http://jtxie.com/models/mpncovresnet50-15991845.pth',
    'mpncovresnet101': 'http://jtxie.com/models/mpncovresnet101-ade9737a.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


featurepath = r"excel/feature.xlsx"
labelpath = r"excel/label.xlsx"
predictpath = r"excel/predict.xlsx"
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='Resnet50_18_Market1501', type=str, help='output model name')
# parser.add_argument('--data_dir',default='/home/mcii216/fmx/dataset_ReID/Market-1501/pytorch',type=str, help='training dir path')
parser.add_argument('--data_dir',default='/home/mcii216/fmx/dataset_ReID/DukeMTMC-reID/pytorch',type=str, help='training dir path')
# parser.add_argument('--data_dir',default='/home/mcii216/fmx/dataset_ReID/cuhk03/detected/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--ft_net', action='store_true', help='use Resnet50' )
parser.add_argument('--ft_net_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--Inceptionv3', action='store_true', help='use Inception_v3' )
parser.add_argument('--VGG16', action='store_true', help='use VGG16' )
parser.add_argument('--twostream_Resnet50', action='store_true', help='use twostream_Resnet50' )
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((384,192)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list + [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
#################################################################
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs, pretrained = True):
    since = time.time()
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # if pretrained:
    #     if opt.resume:
    #         if os.path.isfile(os.path.join('./model/twostream_Resnet50_Market1501', opt.resume)):
    #             checkpoint = torch.load(os.path.join('./model/twostream_Resnet50_Market1501', opt.resume))
    #             pretrained_dict = checkpoint['state_dict']
    #             model_dict = model.state_dict()
    #             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #             model_dict.update(pretrained_dict)
    #             model.load_state_dict(model_dict)

    best_acc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                if fp16:
                    inputs = inputs.half()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.ft_net or opt.ft_net_dense or opt.Inceptionv3:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                elif opt.twostream_Resnet50:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 7

                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5]) + sm(
                        part[6])
                    _, preds = torch.max(score.data, 1) #preds=[524, 114, 114, 524, 114........]

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)
                elif opt.PCB:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                else:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)



                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                is_best = epoch_acc > best_acc
                best_acc = max(epoch_acc, best_acc)
                if best_acc == epoch_acc:
                    best_epoch = epoch
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        filename = []
        filename.append(os.path.join('./model', name, 'checkpoint.pth.tar'))
        filename.append(os.path.join('./model', name, 'model_best.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.name,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(last_model_wts)
    # save_network(model, 'last')
    return model,


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


######################################################################
# Finetuning the convnet
# Load a pretrainied model and reset final fully connected layer.

if opt.ft_net_dense:
    model = ft_net_dense(len(class_names), opt.droprate)
elif opt.twostream_Resnet50:
    model = twostream_Resnet50(len(class_names))
elif opt.PCB:
    model = PCB(len(class_names))
elif opt.VGG16:
    model = VGG16(len(class_names))
elif opt.Inceptionv3:
    model = Inceptionv3(len(class_names))
elif opt.Resnet101:
    model=Resnet101(len(class_names))
else:
    model = ft_net(len(class_names), opt.droprate)
print(model)

if opt.twostream_Resnet50:
    ignored_params1 = list(map(id, model.classifier.parameters()))
    ignored_params2 = (list(map(id, model.classifier0.parameters()))
                        + list(map(id, model.classifier1.parameters()))
                        + list(map(id, model.classifier2.parameters()))
                        + list(map(id, model.classifier3.parameters()))
                        + list(map(id, model.classifier4.parameters()))
                        + list(map(id, model.classifier5.parameters()))
                        )

    base_params1 = filter(lambda p: id(p) not in ignored_params1, model.submodel1.parameters())
    base_params2 = filter(lambda p: id(p) not in ignored_params2, model.submodel2.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params1, 'lr':  opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr},
        {'params': base_params2, 'lr':  opt.lr},
        {'params': model.classifier0.parameters(), 'lr': opt.lr},
        {'params': model.classifier1.parameters(), 'lr': opt.lr},
        {'params': model.classifier2.parameters(), 'lr': opt.lr},
        {'params': model.classifier3.parameters(), 'lr': opt.lr},
        {'params': model.classifier4.parameters(), 'lr': opt.lr},
        {'params': model.classifier5.parameters(), 'lr': opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

elif opt.PCB:
    # ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params = (list(map(id, model.classifier0.parameters() ))
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': opt.lr},
             # {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

elif opt.VGG16:
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': opt.lr},

             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

elif opt.Inception_v3:
    ignored_params =list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': opt.lr},
        # {'params': model.fc.parameters(), 'lr': opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

model = model.cuda()

######################################################################
if opt.resume:
    if os.path.isfile(os.path.join('./model', name, opt.resume)):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(os.path.join('./model',name, opt.resume))
        opt.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

######################################################################
criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100, pretrained = True)

