# **Behavioral Cloning** 

## Writeup Template

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

My model consists of a convolution neural network the same as the nivida model

![Screenshot from 2022-01-07 21-28-00](https://user-images.githubusercontent.com/8016115/148632741-d4bf02ab-9ba9-49e5-b48e-01827e85a767.png)
#### 2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

IMage is fliped and steering is negated to add more training data 
left/right image is also used as well to avoid overfitting 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Final Model Architecture

The final model architecture is the same as the nvidia model
Here is a visualization of the architecture 


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust the steering when it is shifted to the edge. To augment the data sat, I also flipped images and angles.I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10
