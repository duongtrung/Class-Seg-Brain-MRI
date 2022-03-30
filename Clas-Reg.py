import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize

import os
# To disable all logging output from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from skimage import io
import os
import glob
import random
import time

data_path = "" # download the dataset from https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
result_path = "./"

brain_df = pd.read_csv(data_path + 'data_mask.csv')
brain_df_train = brain_df.drop(columns = ['patient_id'])

# Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))

# split the data into train and test data
from sklearn.model_selection import train_test_split
train, test = train_test_split(brain_df_train, test_size = 0.15)

from keras_preprocessing.image import ImageDataGenerator
# Create a data generator which scales the data from 0 to 1 and makes validation split of 0.15
datagen = ImageDataGenerator(rescale=1./255., validation_split = 0.15)

train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= data_path,
x_col='image_path',
y_col='mask',
subset="training",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(256,256))


valid_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= data_path,
x_col='image_path',
y_col='mask',
subset="validation",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(256,256))

# Create a data generator for test images
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=test,
directory= data_path,
x_col='image_path',
y_col='mask',
batch_size=16,
shuffle=False,
class_mode='categorical',
target_size=(256,256))


#################################################################################
# model zoos: ResNet50, ResNet50V2, InceptionResNetV2, EfficientNetB0, EfficientNetB7, MobileNetV2
current_model = "EfficientNetB7"
if current_model == "ResNet50":
    basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
if current_model == "ResNet50V2":
    basemodel = ResNet50V2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
if current_model == "InceptionResNetV2":
    basemodel = InceptionResNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
if current_model == "EfficientNetB0":
    basemodel = EfficientNetB0(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
if current_model == "EfficientNetB7":
    basemodel = EfficientNetB7(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
if current_model == "MobileNetV2":
    basemodel = MobileNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))

# freeze the model weights
for layer in basemodel.layers:
    layers.trainable = False

# Add classification head to the base model
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (2,2))(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(1024, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(1024, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(1024, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(2, activation = 'softmax')(headmodel)

model = Model(inputs = basemodel.input, outputs = headmodel)

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])

# use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=30)

# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath= data_path + "classifier-" + current_model + "-weights.hdf5", verbose=2, save_best_only=True)

################################################################################
f = open(result_path + current_model + "log.txt", "w")

start_time = time.time()
history = model.fit(train_generator, steps_per_epoch= train_generator.n // 16, epochs = 20, validation_data= valid_generator, validation_steps= valid_generator.n // 16, callbacks=[checkpointer, earlystopping])
duration = time.time() - start_time
print(f"duration {duration}")

# save the model architecture to json file for future use
model_json = model.to_json()
with open(result_path + "classifier-architecture-" + current_model + "-model.json","w") as json_file:
  json_file.write(model_json)


# make prediction
test_predict = model.predict(test_generator, steps = test_generator.n // 16, verbose =2)

# Obtain the predicted class from the model prediction
predict = []

for i in test_predict:
  predict.append(str(np.argmax(i)))

predict = np.asarray(predict)

np.savetxt(result_path + current_model + "_prediction.txt", predict, encoding="utf8", fmt = '%s')

# since we have used test generator, it limited the images to len(predict), due to batch size
original = np.asarray(test['mask'])[:len(predict)]

# Obtain the accuracy of the model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(original, predict)
print(f"accuracy {accuracy}")

# plot the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(original, predict)
#plt.figure(figsize = (7,7))
#sns.heatmap(cm, annot=True)

import itertools

def plot_confusion_matrix(cm, list_of_range_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Remember to import: from sklearn.metrics import confusion_matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(list_of_range_classes))
    plt.xticks(tick_marks, list_of_range_classes, rotation=45)
    plt.yticks(tick_marks, list_of_range_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm, list(range(2)))

from sklearn.metrics import classification_report

report = classification_report(original, predict, labels = [0,1])
print("classification_report\n")
print(report)

