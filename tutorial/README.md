# Twostream-Person-reID-Practical
By [Meixia Fu]

This is a [Beijing University of Posts and Telecommunications](https://www.bupt.edu.cn/) computer vision practical, authored by Meixia Fu.
The practical explores the basis of learning pedestrian features. In this practical, we will learn to build a simple person re-ID system step by step. **Any suggestion is welcomed.**


Person re-ID aims to identify the pedestrian from other different cameras through establishing correspondence with a given image or video of the same person taken from one camera.

## Prerequisites
- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+ (http://pytorch.org/)
- Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

## Getting started
Check the Prerequisites. The download links for this practice are:

- Code: [Practical-Baseline](https://github.com/Prettyfinger/Twostream_reID)
- Data: [Market-1501](http://blog.fangchengjin.cn/reid-market-1501.html)
- Data: [DukeMTMC-re-ID](http://blog.fangchengjin.cn/reid-duke.html)
- Data: [CUHK03](http://blog.fangchengjin.cn/reid-cuhk03.html)

## Part 1: Training
### Part 1.1: Prepare Data Folder (`python prepare.py`)https://github.com/Prettyfinger/Twostream_reID
You may notice that the downloaded folder is organized as:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```
Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path, such as `\home\zzd\Download\Market`. Run this script in the terminal.
```bash
python prepare.py
```
We create a subfolder called `pytorch` under the download folder. 
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
```

In every subdir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 

### Part 1.2: Build Neural Network (`model.py`)
We can use the pretrained networks, such as InceptionNet`, `VGG16`, `ResNet` and `DenseNet`. Generally, the pretrained networks help to achieve a better performance, since it preserves some good visual patterns from ImageNet [1].

In pytorch, we can easily import them by two lines. For example,
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```
You can simply check the structure of the model by:
```python
print(model)
```

But we need to modify the networks a little bit. There are 751 classes (different people) in Market-1501, which is different with 1,000 classes in ImageNet. So here we change the model to use our classifier.

```python
import torch
import torch.nn as nn
from torchvision import models

# Define the twostream_Resnet50 Model
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


```diff
+ Quick Question. Why we use AdaptiveAvgPool2d? What is the difference between the AvgPool2d and AdaptiveAvgPool2d?
+ Quick Question. Does the model have parameters now? How to initialize the parameter in the new layer?
```
More details are in `model.py`. You may check it later after you have gone through this practical.

### Part 1.3: Training (`python train.py`)
OK. Now we have prepared the training data and defined model structure.

We can train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of the model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Let's look at what we do in the `train.py`.
The first thing is how to read data and their labels from the prepared folder.
Using `torch.utils.data.DataLoader`, we can obtain two iterators `dataloaders['train']` and `dataloaders['val']` to read data and label.
```python
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```

Here is the main code to train the model.
Yes. It's only about 20 lines. Make sure you can understand every line of the code.
```python
            # Iterate over data.
            for data in dataloaders[phase]:
                # get a batch of inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable, if gpu is used, we transform the data to cuda.
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                #-------- forward --------
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #-------- backward + optimize -------- 
                # only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
```
```diff
+ Quick Question. Why we need optimizer.zero_grad()? What happens if we remove it?
+ Quick Question. The dimension of the outputs is batchsize*751. Why?
```
Every 10 training epoch, we save a snapshot and update the loss curve.
```python
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
```

## Part 2: Test
### Part 2.1: Extracting feature (`python test.py`)
In this part, we load the network weight (we just trained) to extract the visual feature of every image.
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--name` the dir name of the trained model.


`--batchsize` batch size.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.

Let's look at what we do in the `test.py`.
First, we need to import the model structure and then load the weight to the model.
```python
model_structure = ft_net(751)
model = load_network(model_structure)
```
For every query and gallery image, we extract the feature by simply forward the data.
```python
outputs = model(input_img) 
# ---- L2-norm Feature ------
ff = outputs.data.cpu()
fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
ff = ff.div(fnorm.expand_as(ff))
```
```diff
+ Quick Question. Why we flip the test image horizontally when testing? How to fliplr in pytorch?
+ Quick Question. Why we L2-norm the feature?
```

### Part 2.2: Evaluation
Yes. Now we have the feature of every image. The only thing we need to do is matching the images by the feature.
```bash
python evaluate_gpu.py
```

Let's look what we do in `evaluate_gpu.py`. We sort the predicted similarity score.
```python
query = qf.view(-1,1)
# print(query.shape)
score = torch.mm(gf,query) # Cosine Distance
score = score.squeeze(1).cpu()
score = score.numpy()
# predict index
index = np.argsort(score)  #from small to large
index = index[::-1]
```

Note that there are two kinds of images we do not consider as right-matching images.
* Junk_index1 is the index of mis-detected images, which contain the body parts.

* Junk_index2 is the index of the images, which are of the same identity in the same cameras.

```python
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    # The images of the same identity in different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Only part of body is detected. 
    junk_index1 = np.argwhere(gl==-1)
    # The images of the same identity in same cameras
    junk_index2 = np.intersect1d(query_index, camera_index)
```

We can use the function `compute_mAP` to obtain the final result.
In this function, we will ignore the junk_index.
```python
CMC_tmp = compute_mAP(index, good_index, junk_index)
```

## Part 3: A simple visualization (`python demo.py`)
To visualize the result, 
```
python demo.py --query_index 777
```
`--query_index ` which query you want to test. You may select a number in the range of `0 ~ 3367`.

It is similar to the `evaluate.py`. We add the visualization part.
```python
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10): #Show top-10 images
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green') # true matching
        else:
            ax.set_title('%d'%(i+1), color='red') # false matching
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
```

## Part 4: Your Turn. 

- Market-1501 is a dataset collected at Tsinghua University in summer.

Let's try another dataset called DukeMTMC-reID, which is collected at Duke University in winter.

You may download the dataset at [Here](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip). Try it by yourself.

The dataset is quite similar to Market-1501. You may also check with the state-of-the-art results at [Here](https://github.com/layumi/DukeMTMC-reID_evaluation/tree/master/State-of-the-art). 

```diff
+ Quick Question. Could we directly apply the model trained on Market-1501 to DukeMTMC-reID? Why?
```


## Part5: Other Related Works
- The pedestrian has some specific attributes, e.g., gender, carrying. They can help the feature learning. We annotate the ID-level attributes for Market-1501 and DukeMTMC-reID. You could check [this paper](https://arxiv.org/abs/1703.07220).
![](https://github.com/vana77/DukeMTMC-attribute/blob/master/sample_image.jpg)

- Could we use natural language as query? Check [this paper](https://arxiv.org/abs/1711.05535).
![](https://github.com/layumi/Image-Text-Embedding/blob/master/CUHK-show.jpg)

- Could we use other losses (i.e. contrastive loss) to further improve the performance? Check [this paper](https://arxiv.org/abs/1611.05666). 

- Person-reID dataset is not large enough to train a deep-learned network? You may check [this paper](https://arxiv.org/abs/1701.07717) and try some data augmentation method like [random erasing](https://arxiv.org/abs/1708.04896).

- Pedestrian detection is bad? Try [Open Pose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [Spatial Transformer](https://github.com/layumi/Pedestrian_Alignment) to align the images.

## Reference

[1] 

