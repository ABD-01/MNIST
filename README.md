# MNIST Digit Classifier

#### Implemented 2 types ML Model on [The MNIST Database](http://yann.lecun.com/exdb/mnist/)

# 1. [Single Layer NN](https://github.com/ABD-01/MNIST/tree/main/Single%20Layer%20Model)
####  Take input with input features = 28*28. Uses a Single Layer(also the output layer) to implement the model.
  ![Cost vs No. of Iterations](https://github.com/ABD-01/MNIST/blob/main/Single%20Layer%20Model/SingleCost.png)
####  `Learning Rate = 0.009`
  
####  `Accuracy on Test Set = 94.181 %`

# 2. [Multi Layer NN](https://github.com/ABD-01/MNIST/tree/main/Multi%20Layer%20Model)
####  Uses 4 Layers to train the model which takes 28*28 input features.
####  Nodes per Layer are:
#####  n<sup>[1]</sup> = 512
#####  n<sup>[2]</sup> = 128
#####  n<sup>[3]</sup> = 32
#####  n<sup>[4]</sup> = 10

  ![Cost vs No. of Iterations](https://github.com/ABD-01/MNIST/blob/main/Multi%20Layer%20Model/MultiCost.png)
#### `Learning Rate = 0.11`
#### `Accuracy on Train Set = 98.272 %`
#### `Accuracy on Test  Set = 98.259 %`
