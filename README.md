# TGNet  
Image Semantic Segmentation Based on Graph Convolution Nerual Network

## TGNet and TGModule(A GCN Module, based on GloRe)

### TGMoudle(based on GloRe)

<img width="80%" height="80%" align=center src="https://github.com/Junelin2333/TGNet/blob/main/misc/TGModule.png" alt="Figure 1: TGMoudle" />

### TGNet

<img width="60%" height="60%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/TGN.png" alt="Figure 2: TGNet" align=center />

### GloRe-Net(as baseline, only use for comparing with our method)

<img width="40%" height="40%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/GloRe-Net.png" alt="Figure 3: GloRe-Net" align=center />

## Enviroment

- Tensorflow 2.3
- Python 3.8.5
- CUDA 10.1
- tensorflow-addons latest version
- and so on ...

## Training

run train.py and remenber to change the parsers

## Result

### Result on ADE20k Dataset

#### visulizaiton

<img width="60%" height="60%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/%E6%88%AA%E5%B1%8F2021-06-05%2012.38.06.png" alt="Figure 4" align=center />

#### Metrics

<img width="80%" height="80%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/%E6%88%AA%E5%B1%8F2021-06-05%2012.38.49.png" alt="Figure 5" align=center />


### Result on Pascal Context Dataset

#### visulizaiton

<img width="60%" height="60%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/%E6%88%AA%E5%B1%8F2021-06-05%2012.39.04.png" alt="Figure 6" align=center />

#### Metrics

<img width="80%" height="80%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/%E6%88%AA%E5%B1%8F2021-06-05%2012.39.22.png" alt="Figure 7" align=center />

#### Addon Result: Numbers of Node and Different Backbone

<img width="80%" height="80%" src="https://github.com/Junelin2333/TGNet/blob/main/misc/%E6%88%AA%E5%B1%8F2021-06-05%2012.39.48.png" alt="Figure 8" align=center />
