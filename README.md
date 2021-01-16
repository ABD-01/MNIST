# Kaggle Digit Recognizer
This is [my submission](https://www.kaggle.com/abd931/kaggel-digit-recognizer) to the Kaggle's [Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer/)

#### My Model comprises of two Convolutional Layers followed by three Fully Connected Layers.

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
  <img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6509800%2F298fcb3379ca760dcab33be1c13084bc%2Fkaggle%20digit%20recognizer%20learning%20curve.png?generation=1610788352058574&alt=media' alt='Cost vs No. of Iterations' width='840'>

```css
  Batch size = 100
  Learning Rate = 0.002
  iterations = 50
  Accuracy on Train Set = 99.819 %
  Accuracy on Test  Set = 98.32 %
```

