# DEFNet

# 
## Deep Ensemble Feature Network for Gastric Section Classification

![image](https://github.com/nchucvml/DEFNetwork/blob/main/flowchart.PNG)

## Code coming soon

Abstract — In this paper, we propose a novel deep ensemble feature (DEF) network to classify gastric sections from endoscopic images. Different from recent deep ensemble learning methods, which need to train deep features and classifiers individually to obtain fused classification results, the proposed method can simultaneously learn the deep ensemble feature from arbitrary number of convolutional neural networks (CNNs) and the decision classifier in an end-to-end trainable manner. It comprises two sub networks, the ensemble feature network and the decision network. The former sub network learns the deep ensemble feature from multiple CNNs to represent endoscopic images. The latter sub network learns to obtain the classification labels by using the deep ensemble feature. Both sub networks are optimized based on the proposed ensemble feature loss and the decision loss which guide the learning of deep features and decisions. As shown in the experimental results, the proposed method outperforms the state-of-the-art deep learning, ensemble learning, and deep ensemble learning methods.

## Reference
* If you use the DEF network for your research, please cite the following papers.

[1] T.-H. Lin, J.-Y. Jhang, C.-R. Huang, Y.-C. Tsai, H.-C. Cheng and B.-S. Sheu, "Deep Ensemble Feature Network for Gastric Section Classification,"  IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 1, pp. 77-87, Jan. 2021, doi: 10.1109/JBHI.2020.2999731.

[2] T.-H. Lin, C.-R. Huang, H.-C. Cheng and B.-S. Sheu, "Gastric section detection based on decision fusion of convolutional neural networks", in Proc. IEEE Biomed. Circuits Syst. Conf., pp. 1-4, Oct. 2019.

### Due to our medical dataset cannot be released, we utilize an open source dataset for demonstration.
### Important Note: DEF Network is suitable for higher resolution image datasets.

## Demo
Dataset: dollars image (250x120)

source: https://www.kaggle.com/nsojib/bangla-money

###### Create tfrecord (we have already create the tfrecord of dollars in the ./tfrecord)


Place your training data in ./trainall folder as  ./trainall/1/1_0.jpg, 1_1.jpg ...
																								./trainall/2/2_0.jpg, 2_1.jpg ...
																								./trainall/5/5_0.jpg, 5_1.jpg ...
																										.
																										.
																										.

Modify the ./datasets/convert_data.py _NUM_VALIDATION = 333 
(333 means that it will randomly choose 100 images as your testing data, and the other data will be your training data)

```
python download_and_convert_data.py  --dataset_name=myown --dataset_dir=./
```

###### Place tfrecord

Put the data_train_00000-of-00001.tfrecord to ./tfrecord/
Put the data_validation_00000-of-00001.tfrecord to ./tfrecord/
Put the labels.txt to ./tfrecord/

###### Training

Modify the ./datasets/decode_tfrecord.py SPLITS_TO_SIZES = {'train': 1637, 'validation:' 333}

Modify the ./datasets/decode_tfrecord.py _NUM_CLASSES = 9

(1637 and 333 means that the numbers of your training data and testing data, respectively. Please modify the numbers by yourself.)

```
python main.py
```

###### Testing

Modify the ./main.py

Comment out line 379

Uncomment   line 380

```
377	if __name__ == '__main__':
378	    tf.reset_default_graph()
379	    # Train()
380	    Test()
```

```
python main.py
```


##############################################

###### Environment settings

Pretrained weights of VGG19, Inception_v3, Resnet v2_50 are from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

Pretrained weights of Densenet121 are from https://github.com/pudae/tensorflow-densenet


The most important packages:

```
tensorfow					1.14.0
tensorflow-gpu				1.14.0
cuda						10.0
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
