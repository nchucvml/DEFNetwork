# DEFNet

# 
## Deep Ensemble Feature Network for Gastric Section Classification

DEFNet is a novel end-to-end trainable deep ensemble neural network for general classification tasks.

In this project, we demonstrate the performance of the DEFNet for gastric section classification and dollars image classification.


![image](https://github.com/nchucvml/DEFNetwork/blob/main/flowchart.PNG)

Abstract — In this paper, we propose a novel deep ensemble feature (DEF) network to classify gastric sections from endoscopic images. Different from recent deep ensemble learning methods, which need to train deep features and classifiers individually to obtain fused classification results, the proposed method can simultaneously learn the deep ensemble feature from arbitrary number of convolutional neural networks (CNNs) and the decision classifier in an end-to-end trainable manner. It comprises two sub networks, the ensemble feature network and the decision network. The former sub network learns the deep ensemble feature from multiple CNNs to represent endoscopic images. The latter sub network learns to obtain the classification labels by using the deep ensemble feature. Both sub networks are optimized based on the proposed ensemble feature loss and the decision loss which guide the learning of deep features and decisions. As shown in the experimental results, the proposed method outperforms the state-of-the-art deep learning, ensemble learning, and deep ensemble learning methods.

Keyword - Deep ensemble learning, ensemble learning, deep learning, endoscopic image

## Results
The gastric section classification results of [1]. 

![image](https://github.com/nchucvml/DEFNetwork/blob/main/experiments.PNG)

Because our medical dataset cannot be released, we utilize an open source dataset for demonstration in the following.

### Important Note: DEF Network is suitable for higher resolution (at least hundreds by hundreds) image datasets.

## Reference
If you use the DEF network for your research, please cite the following papers.

[1] T.-H. Lin, J.-Y. Jhang, C.-R. Huang, Y.-C. Tsai, H.-C. Cheng and B.-S. Sheu, "Deep Ensemble Feature Network for Gastric Section Classification,"  IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 1, pp. 77-87, Jan. 2021, doi: 10.1109/JBHI.2020.2999731.

[2] T.-H. Lin, C.-R. Huang, H.-C. Cheng and B.-S. Sheu, "Gastric section detection based on decision fusion of convolutional neural networks", in Proc. IEEE Biomed. Circuits Syst. Conf., pp. 1-4, Oct. 2019.

## Usage
Dataset: dollars image (250x120)

source: https://www.kaggle.com/nsojib/bangla-money

Prtrained models: 

Pretrained weights of VGG19, Inception_v3, Resnet v2_50 are from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

Pretrained weights of Densenet121 are from https://github.com/pudae/tensorflow-densenet

### Create tfrecord

Place your training data in the ./dollars folder.

```
python download_and_convert_data.py
```

The tfrecord of dollars will be generated in the ./tfrecord.

You can modify the default argument values for your own dataset. 
```
default augment value:

--dataset_name=myown
--dataset_source_train_dir=./dollars/train
--dataset_source_test_dir=./dollars/test
--dataset_destination_dir=./tfrecord
```

### Training

```
python main.py --mode train
```

```
default augument value:

# input param
--source_data_dir './tfrecord'
--num_classes 9
--num_train 1637
--num_test 333

# output param
--weights_dir ./result
--logs_dir ./logs
--logits_dir ./logits

# training param
--training_batch_size 16    
--training_learning_rate 1e-5
--training_epochs 500
```
We suggest a larger training batch size! 

### Testing

```
python main.py --mode test
```

```
default augment value:

# testing param
--testing_your_pretrained_weights_path ./result/fine_tune.ckpt-500
--testing_your_dataset_labesls ['1', '2', '5', '10', '20', '50', '100', '500', '1000']
```

----------------------------------

## Environment settings

The most important packages:

```
tensorfow					1.14.0
tensorflow-gpu		1.14.0
cuda						  10.0
```

The other detail packages:

```
absl-py                     0.13.0
appdirs                     1.4.4
argon2-cffi                 20.1.0
astor                       0.8.1
async-generator             1.10
attrs                       20.3.0
backcall                    0.2.0
bleach                      3.2.3
cached-property             1.5.2
certifi                     2020.12.5
cffi                        1.14.4
chardet                     4.0.0
colorama                    0.4.4
cycler                      0.10.0
dataclasses                 0.8
decorator                   4.4.2
defusedxml                  0.6.0
dukpy                       0.2.3
entrypoints                 0.3
future                      0.18.2
gast                        0.2.2
google-pasta                0.2.0
graphviz                    0.16
grpcio                      1.39.0
h5py                        3.1.0
idna                        2.10
importlib-metadata          2.1.1
ipykernel                   5.3.4
ipython                     7.16.1
ipython-genutils            0.2.0
ipywidgets                  7.6.3
javascripthon               0.11
jedi                        0.17.2
Jinja2                      2.11.2
joblib                      1.0.0
jsonpatch                   1.32
jsonpointer                 2.1
jsonschema                  3.2.0
jupyter                     1.0.0
jupyter-client              6.1.7
jupyter-console             6.2.0
jupyter-core                4.7.0
jupyter-echarts-pypkg       0.1.2
jupyterlab-pygments         0.1.2
jupyterlab-widgets          1.0.0
Keras-Applications          1.0.8
Keras-Preprocessing         1.1.2
kiwisolver                  1.3.1
lml                         0.0.2
macropy3                    1.1.0b2
Markdown                    3.3.4
MarkupSafe                  1.1.1
matplotlib                  3.3.3
mistune                     0.8.4
nbclient                    0.5.1
nbconvert                   6.0.7
nbformat                    5.1.2
nest-asyncio                1.4.3
notebook                    6.2.0
numpy                       1.19.5
opencv-python               4.5.1.48
packaging                   20.8
pandas                      1.1.5
pandocfilters               1.4.3
parso                       0.7.1
pickleshare                 0.7.5
Pillow                      8.1.0
pip                         20.3.3
prometheus-client           0.9.0
prompt-toolkit              3.0.8
protobuf                    3.17.1
pycparser                   2.20
pyecharts                   0.5.11
pyecharts-javascripthon     0.0.6
pyecharts-jupyter-installer 0.0.3
pyecharts-snapshot          0.2.0
pyee                        8.1.0
Pygments                    2.7.4
pyparsing                   2.4.7
pyppeteer                   0.2.5
pyrsistent                  0.17.3
python-dateutil             2.8.1
pytorch-warmup              0.0.4
pytz                        2020.5
pywin32                     227
pywinpty                    0.5.7
pyzmq                       20.0.0
qtconsole                   5.0.2
QtPy                        1.9.0
requests                    2.25.1
scikit-learn                0.24.1
scipy                       1.5.4
Send2Trash                  1.5.0
setuptools                  51.3.3.post20210118
six                         1.15.0
tensorboard                 1.14.0
tensorboardX                2.2
tensorflow-estimator        1.14.0
tensorflow-gpu              1.14.0
termcolor                   1.1.0
terminado                   0.9.2
testpath                    0.4.4
threadpoolctl               2.1.0
torch                       1.7.1+cu110
torchaudio                  0.7.2
torchfile                   0.1.0
torchsummary                1.5.1
torchvision                 0.8.2+cu110
torchviz                    0.0.1
tornado                     6.1
tqdm                        4.61.0
traitlets                   4.3.3
typing-extensions           3.7.4.3
urllib3                     1.26.5
visdom                      0.1.8.9
visdom-logger               0.1
wcwidth                     0.2.5
webencodings                0.5.1
websocket-client            1.0.1
websockets                  8.1
Werkzeug                    2.0.1
wget                        3.2
wheel                       0.36.2
widgetsnbextension          3.5.1
wincertstore                0.2
wrapt                       1.12.1
xlrd                        1.2.0
zipp                        3.4.0
```
