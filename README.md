# MNIST Digit Classifier

#### Implemented ML Model on [The MNIST Database](http://yann.lecun.com/exdb/mnist/) using 3 different approach.

## 1. [Single Layer NN](https://github.com/ABD-01/MNIST/tree/main/Single%20Layer%20Model)
####  Take input with input features = 28*28. Uses a Single Layer(also the output layer) to implement the model.
  ![Cost vs No. of Iterations](https://github.com/ABD-01/MNIST/blob/main/Single%20Layer%20Model/SingleCost.png)
  
```css
Learning Rate = 0.009  
Accuracy on Test Set = 94.181 %
```

---

## 2. [Multi Layer NN](https://github.com/ABD-01/MNIST/tree/main/Multi%20Layer%20Model)
####  Uses 4 Linear Layers to train the model which takes 28*28 input features.

#### Network Architecture:
<img src='https://github.com/ABD-01/MNIST/blob/main/Multi%20Layer%20Model/MultiLayerModel.png' alt='MultiLayerArchitecture' width='784'>

#### Learning Curve:
  ![Cost vs No. of Iterations](https://github.com/ABD-01/MNIST/blob/main/Multi%20Layer%20Model/MultiCost.png)
  
```css
  Learning Rate = 0.11
  Accuracy on Train Set = 98.272 %
  Accuracy on Test  Set = 98.259 %
```

---

## 3. [Convolutional NN](https://github.com/ABD-01/MNIST/tree/main/CNN%20Model)
#### This Network comprises of two Convolutional Layers followed by three Fully Connected Layers.

#### Network Architecture:
<img src='https://github.com/ABD-01/MNIST/blob/main/CNN%20Model/cnn_arch.jpg' alt='CNNArchitecture' width='784'>

```coffeescript
Network [
  Conv1    : [in_channels=1, out_channels=6, kernel_size=5, stride=1],
  MaxPool1 : [kernel_size=2, stride=2],
  Conv2    : [in_channels=6, out_channels=12, kernel_size=5, stride=1],
  MaxPool2 : [kernel_size=2, stride=2],
  FC1      : [in_features=192, out_features=120],
  FC2      : [in_features=120, out_features=60],
  Output   : [in_features=60, out_features=10],
]
```

#### Learning Curve:
  ![Cost vs No. of Iterations](https://github.com/ABD-01/MNIST/blob/main/CNN%20Model/CNN_Cost.png)
  
```css
  Batch size = 100
  Learning Rate = 0.002
  iterations = 50
  Accuracy on Train Set = 99.58 %
  Accuracy on Test  Set = 98.64 %
```

---
