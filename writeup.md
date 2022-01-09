# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md or writeup.pdf summarizing the results


https://user-images.githubusercontent.com/8016115/148632725-50e3b18e-c34e-4b4d-8995-9b69fd443249.mp4


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network the same as the nivida model, the imput size which is 160x320 which is customizable because of training data iamge size. Then all weights of the nvidia model is trained from scratch. The idea is to use transfer learning and utilize a network that is established and have proven to work so that we can train and iterate use our own data. 
![model](https://user-images.githubusercontent.com/8016115/148671689-75e0c4fc-401f-44a8-877b-3a9d5eb36468.png)
![Screenshot from 2022-01-07 21-28-00](https://user-images.githubusercontent.com/8016115/148632741-d4bf02ab-9ba9-49e5-b48e-01827e85a767.png)
#### 2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

IMage is fliped and steering is negated to add more training data 
left/right image is also used as well to avoid overfitting 
Dropout can also be added but it is not included here since the model runs ok.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### DISCUSSION 
### Model Architecture and Training Strategy

#### 1. Final Model Architecture


The final model architecture is the same as the nvidia model. The imput size which is 160x320 which is customizable because of training data iamge size. Then all weights of the nvidia model is trained from scratch. The idea is to use transfer learning and utilize a network that is established and have proven to work so that we can train and iterate use our own data. 
Here is a visualization of the architecture 

![Screenshot from 2022-01-07 21-28-00](https://user-images.githubusercontent.com/8016115/148632741-d4bf02ab-9ba9-49e5-b48e-01827e85a767.png)
![model](https://user-images.githubusercontent.com/8016115/148671689-75e0c4fc-401f-44a8-877b-3a9d5eb36468.png)

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
![1](https://user-images.githubusercontent.com/8016115/148671903-d88fcb94-638d-470e-b23e-b9d6075ec9bd.png)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust the steering when it is shifted to the edge. To augment the data sat, I also flipped images and angles.I finally randomly shuffled the data set and put 20% of the data into a validation set. 

![3](https://user-images.githubusercontent.com/8016115/148671909-cab6f5f4-9b9f-43d3-8f3e-39403b91406c.png)

I also tried to collect some data while driving the oppsite direction of the track just so it can generalize the training data
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
 The ideal number of epochs was 10



### More discussion
1. Potential improvement can be made if we preload Nvidia model's weights which have been trained on huge amount of data then just retrain the last few fully connected layer. In that case we can fully use the power of transfer learning
2. We can also collect more data with different tracks and scene to generalize the model so it wont overfit on this specific loop, For example also use the other track Udacity provided
3. Temporal information is not used in the cnn model, but if we stack a sequence of images and use the temporal information and change the input size of Input for example, we can have a sequence of 5 images, 160x320x15 size of input as training input, potentially will help to improve the robustness
