# Distracted-Driver-Detection
This project used Convolutional neural network to classfity the distracted driver detection with data set of Keggle competition: https://www.kaggle.com/c/state-farm-distracted-driver-detection.

## Data 
Since I used google colab GPU environment, I resized all test,train images and create the X, y to make hdf5 file in my local drive. 
I used color images than gray scale images to capture more momment. Sample code is here:

```
p_safe='C:/Users/user/Downloads/imgs/train/c0/'
dirs = os.listdir(p_safe)

def p_resize():
    for item in dirs:
        if os.path.isfile(p_safe+item):
            im = Image.open(p_safe+item)
            f, e = os.path.splitext(p_safe+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(f+'.jpg', 'JPEG', quality=100)
            
p_resize()
```
I resized images by each classes to get a error immediately if occurs. The sample code of lableing the X and y is inloved in jupyter notebook. 

## Models
I checked linear model first to see how model works and the number of parameters (10*64*64*3). After that, I chekced binary layer model with learning rate and no batch normalizaton. This simple model worked very well with 99% of accuracy.

1. CNN_model(1,2): I add more layers, Batch normalization and dropout to avoid to overfitting. The accuracy was almost 99% with 0.54% of CNN Error.
I wanted to check how model changed due to learning rate, so I checked same model with different learning rate. Based on my model, learning rate with 0.001 performed best.
3. CNN_model_final: I changed higher dense layer to see further. But, the result was not that good as much as first model, I changed layer filters, Dense layer and Maxpooling sizes. I got a little bit better accuracy with 99% and lower CNN Error. So, I used this model to predict test data images.

## Conclusion
My final model worked pretty well in training set. To get further steps, I would like to predict the same number of test set with train set. I also want to try different optimizer like RMSprop instead of Adam and different network model such as VGG-16.
