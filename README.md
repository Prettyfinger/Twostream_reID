## Person_reID_baseline_pytorch


Baseline Code (with bottleneck) for Person-reID (based on [pytorch](https://pytorch.org)).

It is consistent with the new baseline result in several works, e.g., [Deep Fusion Feature Presentations for Nonaligned Person Re-identification](  ) and [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349).

We arrived **Rank@1=93.20%, mAP=79.77%** only with softmax loss. 


Now we have supported:
- twostream_Resnet50
- Multiple Query Evaluation
- Re-Ranking
- Random Erasing
- ResNet/DenseNet/VGG/InceptionNet
- Visualize Training Curves
- Visualize Ranking Result

Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 


## Some News

**What's new:** Visualizing ranking result is added.
```bash
python featuremap.py
pyhon heatmap.py
python file_path.py
pyhon score_rank.py
python train.py
python test.py
python model.py
```

## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
- [ResNet-50]( )  (Rank@1:78.59% mAP:53.33%)
- [Inceptionv3]( ) (Rank@1:65.38% mAP:35.07%)
- [PCB]( ) (Rank@1:92.46% mAP:77.47%)

## Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites
- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- [Optional] apex (for float16) 


## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

## Dataset & Preparation
Download [Market1501 Dataset](http://blog.fangchengjin.cn/reid-market-1501.html)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset](http://blog.fangchengjin.cn/reid-duke.html).
Our code is on DukeMTMC-reID **Rank@1=83.25%, mAP=70.41%**. Hyperparameters are need to be tuned.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset](http://blog.fangchengjin.cn/reid-cuhk03.html).
Our code is on CUHK03-detected **Rank@1=62.50%, mAP=58.03%**. Hyperparameters are need to be tuned.

## Train
Train a model by
```bash
python train.py --gpu_ids 0 --name twostream_Resnet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

## Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


## Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.


## Citation
As far as I know, the following papers may be the first two to use the bottleneck baseline. You may cite them in your paper.
```
@article{DBLP:ECCV/2018,
  author    = {Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang},
  title     = {Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)},
  booktitle   = {ECCV},
  year      = {2018},
}
```

## Related Repos
1. [Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.09349)
2. [Twostream Person re-ID]( )

