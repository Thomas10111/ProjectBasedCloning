from keras.layers import Cropping2D, Conv2D, Dense, GlobalAveragePooling2D, Activation, Flatten, Lambda, Dropout
from keras.models import Sequential, load_model
import keras
import csv
import cv2
import numpy
import sklearn
import os.path
# center, left (steer right), right (steer left) 
correction_angles = [[0.0, 0.2, -0.2], [0.4, 0.6, 0.5], [-0.7, -0.6, -0.9]]


def generator(samples, batch_size=32):
    num_samples = len(samples)
    samples = sklearn.utils.shuffle(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                for i, correction in enumerate(correction_angles[batch_sample[7]]):
                    filename= batch_sample[i].split('/')[-1]
                    current_path='data/IMG/' + filename
                    
                    # In the recorded images the path is absolute
                    if not os.path.exists(current_path):
                        current_path = batch_sample[i] 
                    
                    image = cv2.imread(current_path)
                    if image is not None:
                        #print("Found: ", current_path)
                        images.append(image)
                        measurements.append(float(batch_sample[3])+ correction)
                        images.append(numpy.fliplr(image))
                        measurements.append(-(float(batch_sample[3])+ correction))

                    else:
                        print("None: ", current_path)
                        exit(-1)

            # trim image to only see section with road
            X_train = numpy.array(images)
            y_train = numpy.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        line.extend([0])
        lines.append(line)
    
#     # repeat for bridge
#     for line in lines[84:165]:
#         lines.append(line)

       
with open('data/IMG_left/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)
    for line in reader:
        line.extend([1])
        lines.append(line)
    for line in lines[2807:3190]:
        lines.append(line)
        
with open('data/IMG_right/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)
    for line in reader:
        line.extend([2])
        lines.append(line)  
    for line in lines[1963:2745]:
        lines.append(line)
        
with open('data/IMG_center/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        line.extend([0])
        lines.append(line)        

from sklearn.model_selection import train_test_split
train_samples, validation_samples= train_test_split(lines, test_size=0.2)
 
    
# images =  []
# measurements = []
# for line in lines:
#     for i, correction in enumerate([0.0, 0.3, -0.3]):
#         filename= line[i].split('/')[-1]
#         current_path='data/IMG/' + filename
#         image = cv2.imread(current_path)
#         if image is not None:
#             #print("Found: ", current_path)
#             images.append(image)
#             measurements.append(float(line[3])+ correction)
#             images.append(numpy.fliplr(image))
#             measurements.append(-(float(line[3])+ correction))
            
#         else:
#             print("None: ", current_path)
#             exit(-1)
#              
# y_train = numpy.array(measurements)
# X_train = numpy.stack(images)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

LOAD = True
if LOAD:
    model = load_model('model.h5')
else:
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))
    #model.add(Dropout(0.2))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
    #model.add(Dropout(0.2))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
    #model.add(Dropout(0.15))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    #model.add(Dropout(0.15))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

model.summary()

# Compile the model
# model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.compile(loss='mse', optimizer='adam')
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'))
model.fit_generator(train_generator,
            steps_per_epoch=numpy.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=numpy.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1, callbacks=callbacks)

model.save("model.h5")