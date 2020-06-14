Plot_Hist = False

if not Plot_Hist:
    from keras.layers import Cropping2D, Conv2D, Dense, GlobalAveragePooling2D, Activation, Flatten, Lambda, Dropout
    from keras.models import Sequential, load_model
    import keras
else:
    from matplotlib import pyplot as plt
import csv
import cv2
import numpy
import sklearn
import os.path
import random
import math
import pathlib
import os


# shift image in x then in y direction and scale steering angle
def trans_image_x(image, steer_angle, correction):
    rows, cols, depth = image.shape
    x = random.randint(-20, 20)
    y = random.randint(-20, 20)
    
    M = numpy.float32([[1, 0, x],[0, 1, 0]])        #y = 0
    dst_x = cv2.warpAffine(image, M, (cols,rows))
    
    M = numpy.float32([[1, 0, x],[0, 1, y]])        # shift in x and y direction
    dst_xy = cv2.warpAffine(image, M, (cols,rows))
    
    steering_correction = float(steer_angle) + correction + x/200.0
    
    return dst_x, dst_xy, steering_correction

    

# center, left (steer right), right (steer left) 
correction_angles = [[0.0, 0.2, -0.2], [0.5, 0.6, 0.4], [-0.5, -0.4, -0.6]]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    print("num_samples: ", num_samples)
    while 1: # Loop forever so the generator never terminates
#    for i in range(1):  # to plot histogram, comment out while
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                if abs(float(batch_sample[3])) < 0.055 and random.random() < 0.85:
                    continue
                for i, correction in enumerate(correction_angles[batch_sample[7]]):
                    filename= batch_sample[i].split('/')[-1]
                    current_path=str(pathlib.Path('data/IMG/' + filename))
                    
                    # In the recorded images the path is absolute
                    if not os.path.exists(current_path):
                        p = pathlib.Path(batch_sample[1])
                        current_path = str(pathlib.Path(*p.parts[4:]))

                    image = cv2.imread(current_path)
                    if image is not None:
                        # append image and steering angle
                        images.append(image)
                        steering_correction = float(batch_sample[3])+ correction
                        measurements.append(steering_correction)

                        images.append(numpy.fliplr(image))
                        measurements.append(-steering_correction)
                        
                        # shift image and flip
                        dst_x, dst_xy, steering_correction = trans_image_x(image, batch_sample[3], correction)
                   
                        images.append(dst_x)
                        measurements.append(steering_correction)
                        
                        images.append(numpy.fliplr(dst_x))
                        measurements.append(-steering_correction) 

                        images.append(dst_xy)
                        measurements.append(steering_correction)
                        
                        images.append(numpy.fliplr(dst_xy))
                        measurements.append(-steering_correction) 

                    else:
                        print("None: ", current_path)
            
            X_train = numpy.array(images)
            y_train = numpy.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


def plot_histogram(lines):
    # Plot histogram
    steering_angles = []
    for b in generator(lines):
        steering_angles.extend(b[1])

    #plt.ioff()
    plt.hist(steering_angles, bins=40)
    plt.savefig("hist.png")
    plt.show()
    exit(0)



lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)

    # append all rounds of track
    for i, line in enumerate(reader):
        line.extend([0])
        lines.append(line)
        

# extra images, car driving on left side of the track
with open('data/IMG_left/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        line.extend([1])
        lines.append(line)


# extra images, car driving on right side of the track    
with open('data/IMG_right/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        line.extend([2])
        lines.append(line)
        
# extra images, car on left side of the bridge
with open('data/IMG_bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        line.extend([1])
        lines.append(line)


if Plot_Hist:
    plot_histogram(lines)



from sklearn.model_selection import train_test_split
train_samples, validation_samples= train_test_split(lines, test_size=0.15)

# Set our batch size
batch_size=64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# LOAD = True, loads previous model.h5 file
# LOAD = False, starts new model
LOAD = False
if LOAD:
    print("loading model")
    model = load_model('model.h5')
else:
    try:
        os.remove("model.h5")
    except:
        pass
    
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (20, 20))))
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) # "normalization after cropping", does it make a difference? Smaller image?
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))         # "You can try to use ELU or Leaky ReLU as activation functions." Interesting, difference?
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Conv2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(100))
    model.add(Dropout(0.50))
    model.add(Dense(50))
    model.add(Dropout(0.50))
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
            epochs=3, verbose=1, callbacks=callbacks)

model.save("model.h5")
print("Model saved")