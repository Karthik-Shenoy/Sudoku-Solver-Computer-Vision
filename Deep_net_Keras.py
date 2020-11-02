from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras
import LoadData
import numpy as np

(x_train_t, y_train_t), (x_test, y_test) = mnist.load_data()
x_train_char, y_train_char = LoadData.load_data()

x_train = np.concatenate((x_train_char, x_test))
y_train = np.concatenate((y_train_char, y_test))



x_train = x_train.reshape(28572, 28,28,1)
x_test = x_train_t.reshape(60000, 28,28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_train_t, num_classes=10)
####data augmentation########
'''datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

batch_size = 1
train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
test_gen = datagen.flow(x_test, y_test, batch_size=batch_size)'''
#############################
model = Sequential()
model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))


model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 32,  epochs=10)

accuracy = model.evaluate(x_test, y_test, batch_size=32)
print(accuracy)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''x = np.a
model.predict()'''