from keras.layers import Cropping2D, Conv2D, Dense, GlobalAveragePooling2D, Activation, Flatten, Lambda, Dropout
from keras.models import Sequential, load_model
import keras
import csv
import cv2
import numpy
import sklearn
import os.path
import random

# center, left (steer right), right (steer left) 
correction_angles = [[0.0, 0.4, -0.4], [3.5, 3.6, 3.4], [-3.5, -3.4, -3.6]]


def generator(samples, batch_size=32):
    num_samples = len(samples)
    print("num_samples: ", num_samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
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
                        steering_correction = float(batch_sample[3])+ correction
                        measurements.append(steering_correction)
                        #print("current_path: ", current_path, "      steering_correction: ", steering_correction)

                        images.append(numpy.fliplr(image))
                        measurements.append(-steering_correction)
                        
                        rows,cols, depth = image.shape
                        x = random.randint(-20, 20)
                        M = numpy.float32([[1, 0, x],[0, 1, 0]])
                        dst = cv2.warpAffine(image, M, (cols,rows))
                        images.append(dst)
                        steering_correction = float(batch_sample[3])+ correction
                        measurements.append(steering_correction)
                        
                        images.append(numpy.fliplr(dst))
                        measurements.append(-steering_correction)
                        
                        
                        

                    else:
                        print("None: ", current_path)
                        #exit(-1)

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
        
    # Add bridge images again
    for _ in range(1):
        for line in lines[1744:1826]:   # 82 of 8035
            lines.append(line)
        for line in lines[2573:2657]:   # 84 of 8035
            lines.append(line)
        for line in lines[3408:3517]:   # 109 of 8035
            lines.append(line)
        for line in lines[5208:5291]:   # 83 of 8035
            lines.append(line)
        for line in lines[6037:6123]:   # 83 of 8035
            lines.append(line)
        for line in lines[6867:6954]:   # 83 of 8035
            lines.append(line)
        for line in lines[7692:7784]:   # 83 of 8035
            lines.append(line)

# extra images, car driving on left side of the track           
with open('data/IMG_left/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)
    
    lines_temp = []
    for line in reader:
        line.extend([1])
        lines_temp.append(line)

    # bridge
    for line in lines_temp[2799:3190]:               # 400 of 3285
        lines_temp.append(line)
        
    lines.extend(lines_temp)

# extra images, car driving on right side of the track    
with open('data/IMG_right/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)
    
    lines_temp = []
    for line in reader:
        line.extend([2])
        lines_temp.append(line)   
    
    # bridge
    for line in lines_temp[1963:2745]:
        lines_temp.append(line)
    
    lines.extend(lines_temp)
    
# extra images, car driving on left side of the bridge
for _ in range(4):
    with open('data/IMG_bridge_left/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader, None)

        for line in reader:
            line.extend([1])
            lines.append(line)
        
# extra images, car on left side of the bridge
with open('data/IMG_bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # next(reader, None)
    for line in reader:
        line.extend([1])
        lines.append(line)
        
# with open('data/IMG_center/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     # next(reader, None)
#     for line in reader:
#         line.extend([0])
#     for i in range(2):     
#         for line in lines[2320:2410]:
#             lines.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples= train_test_split(lines, test_size=0.15)
 


# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# LOAD = True, loads previous model.h5 file
# LOAD = False, starts new model
LOAD = True
if LOAD:
    print("loading model")
    model = load_model('model.h5')
else:
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (20, 20))))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.10))
    model.add(Dense(100))
    model.add(Dropout(0.10))
    model.add(Dense(50))
    model.add(Dropout(0.10))
    model.add(Dense(10))
    model.add(Dense(1))

model.summary()

# Compile the model
model.compile(loss='mse', optimizer='adam')
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'))
model.fit_generator(train_generator,
            steps_per_epoch=numpy.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=numpy.ceil(len(validation_samples)/batch_size),
            epochs=1, verbose=1, callbacks=callbacks)

model.save("model.h5")
print("Model saved")