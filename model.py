from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Dense, Flatten, Convolution2D
import matplotlib.pyplot as plt
from random import shuffle

import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from numpy import ceil
import sklearn
from scipy import ndimage
# import np.random.shuffle as shuffle



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #flip image and augmentation
                images.append(cv2.flip(center_image,1))
                angles.append(-center_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []


with open('./data/new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#data augementation by using left and right images
# can also add flip image and negating steering
# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         steering_center = float(row[3])

#         # create adjusted steering measurements for the side camera images
#         correction = 0.2 # this is a parameter to tune
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction

#         # read in images from center, left and right cameras
#         path = "..." # fill in the path to your training IMG directory
#         img_center = process_image(np.asarray(Image.open(path + row[0])))
#         img_left = process_image(np.asarray(Image.open(path + row[1])))
#         img_right = process_image(np.asarray(Image.open(path + row[2])))

#         # add images and angles to data set
#         car_images.extend(img_center, img_left, img_right)
#         steering_angles.extend(steering_center, steering_left, steering_right)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col, ch)))

#model.add(... finish defining the rest of your model architecture here ...)
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)



# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data = 
#     validation_generator,
#     nb_val_samples = len(validation_samples), 
#     nb_epoch=5, verbose=1)

# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()