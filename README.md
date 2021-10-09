# MNIST Digit Classifier

## Table of Content

* [Triplet Network](#4-triplet-nn-with-triplet-loss)
* [Convolutional Network](#3-convolutional-nn)
* [Multi Layer Network (NumPy)](#2-multi-layer-nn-numpy)
* [Single Layer Network (NumPy)](#1-single-layer-nn-numpy)

#### Implemented various models on [The MNIST Database](http://yann.lecun.com/exdb/mnist/) using different approaches to learn new stuff.

## 4. [Triplet NN (with Triplet Loss)](https://github.com/ABD-01/MNIST/blob/main/Triplet%20Loss/TRIPLET_LOSS_Pytorch.ipynb)

Implemented a convolutional network that learns to generate encodings of passed images such as to minimize the triplet loss function given by :

 ℒ(*A*,*P*,*N*) = max( || *f*(*A*)-*f*(*P*) ||<sup>2</sup>) - || *f*(*A*)-*f*(*N*) ||<sup>2</sup> + 𝜶, 0)

where *A* is an anchor input, *P* is a positive input of the same class as *A*, *N* is a negative input of a different class from *A*, 𝜶 is a margin between positive and negative pairs, and *f* is an embedding.

<!-- #### Network Architechture
```json
{
  "name": "Model",
  "arch": {
      "convnet1": {
          "conv1" : "Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2))",
          "conv2" : "Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))",
          "actv" : "ReLU()",
          "pool" : "MaxPool2d(kernel_size=3, stride=2)"
      },
      "convnet2": {
          "conv1" : "Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))",
          "conv2" : "Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))",
          "actv" : "ReLU()",
          "pool" : "MaxPool2d(kernel_size=2, stride=2)"
      },
      "convnet3": {
          "conv1" : "Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))",
          "conv2" : "Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))",
          "actv" : "ReLU()",
          "pool" : "MaxPool2d(kernel_size=3, stride=2)"
      },
      "FullyConnected": {
          "fc1": "Linear(in_features=4096, out_features=2048)",
          "fc2": "Linear(in_features=2048, out_features=512)",
          "fc3": "Linear(in_features=512, out_features=128)"
      } 
  },
  
  "training": {
    "images": "100 images each of classes 0, 1, 2 only",
    "optimizer": "Adam",
    "loss" : "Triplet Loss",
    "batch_size" : 10,
    "epochs" : 5
  },

  "results": {
    "Class 0": { "correct": 5804, "total": 5923,"acuracy": "97.99%" },
    "Class 1": { "correct": 6648, "total": 6742,"acuracy": "98.60%" },
    "Class 2": { "correct": 5830, "total": 5958,"acuracy": "97.85%" },
    "Class 3": { "correct": 5877, "total": 6131,"acuracy": "95.85%" },
    "Class 4": { "correct": 5830, "total": 5842,"acuracy": "99.79%" },
    "Class 5": { "correct": 5274, "total": 5421,"acuracy": "97.28%" },
    "Class 6": { "correct": 5908, "total": 5918,"acuracy": "99.83%" },
    "Class 7": { "correct": 5589, "total": 6265,"acuracy": "89.20%" },
    "Class 8": { "correct": 5777, "total": 5851,"acuracy": "98.73%" },
    "Class 9": { "correct": 5849, "total": 5949,"acuracy": "98.31%" }
  }
}
``` -->
The Network was used to implement *One Shot Learning* which is a technique of learning representations from a single sample.
Images of classes 3 to 9 weren't used while training the model, i.e they were passed to the model for the first time while testing it.

### Training

| Parameter     |                    Value                    |
|---------------|:-------------------------------------------:|
| TrainSet      | 100 images each of `0`, `1` and `2` classes |
| TestSet       |       60,000 images of all ten classes      |
| Loss          |                 Triplet Loss                |
| Learning Rate |                    0.001                    |
| Batch Size    |                      10                     |
| Epochs        |                      5                      |
| Optimizer     |                     Adam                    |

### Results

| Class | Accuracy | Correct | Total |
|:-----:|:--------:|:-------:|:-----:|
|   0   |  97.99%  |   5804  |  5923 |
|   1   |  98.60%  |   6648  |  6742 |
|   2   |  97.85%  |   5830  |  5958 |
|   3   |  95.85%  |   5877  |  6131 |
|   4   |  99.79%  |   5830  |  5842 |
|   5   |  97.28%  |   5274  |  5421 |
|   6   |  99.83%  |   5908  |  5918 |
|   7   |  89.20%  |   5589  |  6265 |
|   8   |  98.73%  |   5777  |  5851 |
|   9   |  98.31%  |   5849  |  5949 |

### Plot
![Cost vs No. of Iterations](Triplet%20Loss/Tripletloss.jpeg)

---

## 3. [Convolutional NN](https://github.com/ABD-01/MNIST/blob/main/CNN%20Model/MNIST_using_CNN_in_pytorch.ipynb)

Trained a Convolutional Neural Network with two layers. Used mini-batches 

### Training

| Parameter     |     Value     |
|---------------|:-------------:|
| TrainSet      |     60,000    |
| TestSet       |     10,000    |
| Loss          | Cross Entropy |
| Learning Rate |     0.002     |
| Batch Size    |      100      |
| Epochs        |       50      |

<!-- ```coffeescript
Network [
  Conv1    : [in_channels=1, out_channels=6, kernel_size=5, stride=1],
  MaxPool1 : [kernel_size=2, stride=2],
  Conv2    : [in_channels=6, out_channels=12, kernel_size=5, stride=1],
  MaxPool2 : [kernel_size=2, stride=2],
  FC1      : [in_features=192, out_features=120],
  FC2      : [in_features=120, out_features=60],
  Output   : [in_features=60, out_features=10],
]
``` -->

### Summary

| Result         |  Value |
|----------------|:------:|
| Train Accuracy | 99.40% |
| Train Correct  |  59641 |
| Test Accuracy  | 98.59% |
| Test Correct   |  9859  |

#### Plot
![Cost vs No. of Iterations](CNN%20Model/CNNCost.jpeg)
![Acc. vs No. of Iterations](CNN%20Model/CNNacc.jpeg)
  
---

## 2. [Multi Layer NN (NumPy)](https://github.com/ABD-01/MNIST/blob/main/Multi%20Layer%20Model/MNIST_Using_Multi_Layer.ipynb)

Trained a Mulit-Layer Neural Net in NumPy.
The model has 4 layers with 512, 128, 32, 10 neurons respectively.

### Training

| Parameter     |     Value     |
|---------------|:-------------:|
| TrainSet      |     60,000    |
| TestSet       |     10,000    |
| Loss          | Cross Entropy |
| Learning Rate |      0.11     |
| Batch Size    |       -       |
| Epochs        |      1000     |

### Summary

| Result         |  Value |
|----------------|:------:|
| Train Accuracy | 98.27% |
| Train Correct  |  57291 |
| Test Accuracy  | 98.26% |
| Test Correct   |  9505  |

### Plot

![Cost vs No. of Iterations](Multi%20Layer%20Model/MultiCost.jpeg)

---

## 1. [Single Layer NN (NumPy)](https://github.com/ABD-01/MNIST/blob/main/Single%20Layer%20Model/MNIST_Single%20Layer.ipynb)

A single layer Neural Net implemented using NumPy library.

### Training

| Parameter     |     Value     |
|---------------|:-------------:|
| TrainSet      |     60,000    |
| TestSet       |     10,000    |
| Loss          | Cross Entropy |
| Learning Rate |     0.009     |
| Batch Size    |       -       |
| Epochs        |      2000     |

### Summary

| Result         |  Value |
|----------------|:------:|
| Train Loss     |  0.50  |
| Train Accuracy | 93.98% |
| Train Correct  |  52507 |
| Test Loss      |  0.80  |
| Test Accuracy  | 94.18% |
| Test Correct   |  8836  |

### Plot

![Cost vs No. of Iterations](Single%20Layer%20Model/SingleCost.jpeg)


---

## Others Branches
* [My solutions to the Assignments of Coursera's course on Deep Learning Specialization](https://github.com/ABD-01/Deep-Learning-Coursera)
* [My submission to the Kaggle's Digit Recognizer Competition](https://github.com/ABD-01/MNIST/tree/kaggle-digit-recognizer)
