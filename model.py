from keras.layers import Cropping2D, Conv2D, Dense, GlobalAveragePooling2D, Activation, Flatten, Lambda
from keras.models import Sequential
import csv
import cv2
import numpy

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path='data/IMG/' + filename
    #print(current_path)
    image = cv2.imread(current_path)
    if image is not None:
        #print("Found: ", current_path)
        images.append(image)
    else:
        print("None: ", current_path)
        exit(-1)
    measurement = float(line[3])
    measurements.append(measurement)

    
y_train = numpy.array(measurements)
X_train = numpy.stack(images)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# Compile the model
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save("model.h5")