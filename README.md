# Distracted-Driver-Detection
This project used Convolutional neural network to classfity the data set of Keggle competition: https://www.kaggle.com/c/state-farm-distracted-driver-detection.

## Data 
Since I used google colab GPU environment, I resized all test,train images and create the X, y to make hdf5 file.
I used color images than gray scale images to capture more momment. 

## Models
I checked linear model first to see how model works and the number of parameters (10*64*64*3). After that, I chekced binary layer model with learning rate and no batch normalizaton. This simple model worked very well with 99% of accuracy.

1. CNN_model(1,2): I add more layers, Batch normalization and dropout to avoid to overfitting. The accuracy was almost 99% with 0.54% of CNN Error.
I wanted to check how model changed due to learning rate, so I checked same model with different learning rate. Based on my model, learning rate with 0.001 performed best.
3. CNN_model_final: I changed higher dense layer to see further. But, the result was not that good as much as first model, I changed layer filters, Dense layer and Maxpooling sizes. I got a little bit better accuracy with 99% and lower CNN Error. So, I used this model to predict test data images.

## Conclusion
My final model worked pretty well. To get further steps, I would like to predict the same number of test set with train set. I also want to try different optimizer like RMSprop and deep convolutional methods use transfer learning on such as VGG-16.
