#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Dropout, MaxPool1D, MaxPool2D, MaxPooling2D, GlobalAvgPool2D, GlobalMaxPool2D, Dropout, Input, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib


# In[ ]:


IMG_SIZE = 176


# In[ ]:


#Referred the following link
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
        zoom_range=[.99, 1.01],
        brightness_range = [0.8, 1.2],
        horizontal_flip=True,
        fill_mode='constant')

data = train_datagen.flow_from_directory('../input/alzheimers-1/Alzheimer_s Dataset/data/', target_size=(IMG_SIZE,IMG_SIZE), subset='training', batch_size=6500)


# In[ ]:


#Splitting the data into 3 different sets, train, test, and validation.
train_data, train_labels = data.next()

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)


# In[ ]:


#Performing SMOTE to balance the dataset.
sm = SMOTE(random_state=42)

train_data_resampled, train_labels_resampled = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)
train_data_resampled = train_data_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# In[ ]:


model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(16, 3, activation='relu', padding='same'),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPool2D(),
        (Conv2D(32, 3, activation='relu', padding='same')),
        (Conv2D(32, 3, activation='relu', padding='same')),
        (BatchNormalization()),
        (MaxPool2D()),
        Conv2D(64, 3, activation='relu', padding='same'),
        (Conv2D(64, 3, activation='relu', padding='same')),
        (BatchNormalization()),
        (MaxPool2D()),
        Conv2D(128, 3, activation='relu', padding='same'),
        (Conv2D(128, 3, activation='relu', padding='same')),
        (BatchNormalization()),
        (MaxPool2D()),
        Dropout(0.2),
        Conv2D(256, 3, activation='relu', padding='same'),
        (Conv2D(256, 3, activation='relu', padding='same')),
        (BatchNormalization()),
        (MaxPool2D()),
        Dropout(0.2),
        Flatten(),
        (Dense(512, activation='relu')),
        (BatchNormalization()),
        (Dropout(0.7)),
        (Dense(128, activation='relu')),
        (BatchNormalization()),
        (Dropout(0.5)),
        (Dense(64, activation='relu')),
        (BatchNormalization()),
        (Dropout(0.3)),
        Dense(4, activation='softmax')        
    ])


# In[ ]:


model.summary()


# In[ ]:


METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='recall')]


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(
    name='Adam',),loss='categorical_crossentropy',metrics=METRICS)


# In[ ]:


model.fit(train_data_resampled, train_labels_resampled, batch_size=32, validation_data=(val_data,val_labels), epochs=150)


# In[ ]:


# joblib.dump(model, 'cnn_model.sav')


# In[ ]:


model.evaluate(test_data, test_labels)


# In[ ]:


#Took this code from the following source
#https://stackoverflow.com/questions/39033880/plot-confusion-matrix-sklearn-with-multiple-labels
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:


classes = ['Mild','Moderate','Non','VeryMild']


# In[ ]:


preds = model.predict(test_data)
confusion_matrix = confusion_matrix(np.argmax(test_labels,axis=1), np.argmax(preds,axis=1))
plot_confusion_matrix(confusion_matrix,classes)


# In[ ]:


from keras.applications.resnet_v2 import ResNet152V2, preprocess_input


# In[ ]:


base = ResNet152V2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False,weights='imagenet')
base.trainable = False


# In[ ]:


resnet = Sequential()
resnet.add(base)
resnet.add(Dropout(0.5))
resnet.add(GlobalAvgPool2D())
resnet.add(BatchNormalization())
resnet.add(Dense(512,activation='relu'))
resnet.add(BatchNormalization())
resnet.add(Dropout(0.7))
resnet.add(Dense(256,activation='relu'))
resnet.add(BatchNormalization())
resnet.add(Dropout(0.5))
resnet.add(Dense(128,activation='relu'))
resnet.add(BatchNormalization())
resnet.add(Dropout(0.5))
resnet.add(Dense(64,activation='relu'))
resnet.add(BatchNormalization())
resnet.add(Dropout(0.5))
resnet.add(Dense(32,activation='relu'))
resnet.add(BatchNormalization())
resnet.add(Dense(4,activation='softmax'))


# In[ ]:


resnet.summary()
resnet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=METRICS)


# In[ ]:


resnet.fit(train_data_resampled, train_labels_resampled, batch_size=32, validation_data=(val_data,val_labels), epochs=150)


# In[ ]:


# pickle.dump(resnet, open('resnet_model.sav', 'wb'))


# In[ ]:


resnet.evaluate(test_data,test_labels)


# In[ ]:


preds = resnet.predict(test_data)

resnet_confusion = confusion_matrix(np.argmax(test_labels,axis=1), np.argmax(preds,axis=1))

plot_confusion_matrix(resnet_confusion,classes)


# In[ ]:


train_reshaped = train_data_resampled.reshape(len(train_data_resampled),-1)
test_reshaped = test_data.reshape(len(test_data),-1)

train_labels_flattened = np.argmax(train_labels_resampled,axis=1)
test_labels_flattened = np.argmax(test_labels,axis=1)


# In[ ]:


rf = RandomForestClassifier(n_estimators=200, criterion='gini',max_depth=None, max_features='auto')
rf.fit(train_reshaped, train_labels_flattened)


# In[ ]:


rf_preds = rf.predict(test_reshaped)
print(accuracy_score(test_labels_flattened, rf_preds))
rf_confusion_matrix = confusion_matrix(test_labels_flattened,rf_preds)


# In[ ]:


plot_confusion_matrix(rf_confusion_matrix,classes)


# In[ ]:


train_labels_with_2_classes = np.where(np.argmax(train_labels,axis=1)==2, 0, 1)
test_labels_with_2_classes = np.where(np.argmax(test_labels,axis=1)==2, 0,1)


# In[ ]:


train_reshaped_2 = train_data.reshape(len(train_data),-1)
test_reshaped_2 = test_data.reshape(len(test_data),-1)
rf2 = RandomForestClassifier(n_estimators=200, criterion='gini',max_depth=None, max_features='auto')
rf2.fit(train_reshaped_2, train_labels_with_2_classes)


# In[ ]:


rf2_preds = rf2.predict(test_reshaped_2)
print(accuracy_score(test_labels_with_2_classes, rf2_preds))
rf_confusion_matrix2 = confusion_matrix(test_labels_with_2_classes,rf2_preds)


# In[ ]:


plot_confusion_matrix(rf_confusion_matrix2,['Non-Demented', 'Demented'])


# In[ ]:




