# loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 

import cv2
from keras.preprocessing import image
import os
from keras.utils import to_categorical
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
import warnings
warnings.filterwarnings("ignore")

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (300,300, 3))

for layer in model.layers:
   layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation="softmax")(x)   

model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.adam(lr=0.0001), metrics=["accuracy"])

data = []
labels = []


for e,i in enumerate(os.listdir('/content/drive/My Drive/wafer')):
  class_path = '/content/drive/My Drive/wafer/'+i
#   print(onlyfiles[0])
  print(class_path)
  for k in os.listdir(class_path):
    im_path = class_path+'/' +str(k)
#     print(im_path)
    img = cv2.imread(im_path)
#     print(img.shape)
    img = cv2.resize(img, (300,300))
#     print(img.shape)
    image = np.array(img)
    data.append(image)
    labels.append(e)

t = np.array(data)
print(t.shape)

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(t, labels, test_size=0.2, stratify = labels)

from keras.utils.np_utils import to_categorical
le = len(np.unique(labels))
y_train = to_categorical(y_train, le)
y_test = to_categorical(y_test, le)

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

train_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range=0.2,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.2,
  width_shift_range = 0.2,
  height_shift_range=0.2,
  rotation_range=20)

test_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range=0.2,
  horizontal_flip = True,
  fill_mode = "nearest",
  zoom_range = 0.2,
  width_shift_range = 0.2,
  height_shift_range=0.2,
  rotation_range=20)
  
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = test_datagen.flow(x_test, y_test, batch_size=32)

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

checkpoint = ModelCheckpoint("/content/drive/My Drive/imagenew_sgd1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

history = model_final.fit_generator(train_generator, steps_per_epoch = x_train.shape[0]//32, 
                              epochs=50, validation_data=val_generator,validation_steps = x_test.shape[0]//32,  
                              verbose=1) 

#callbacks = [checkpoint,early],



  
  
