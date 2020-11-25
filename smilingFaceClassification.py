### Smiling Face Classification - CNN approach.
### Dataset taken from kaggle https://www.kaggle.com/iarunava/happy-house-dataset  Train = 600, Test = 150

# import libraries 
import pandas as pd         
import numpy as np    
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import random

filename = 'train_happy.h5'
f = h5py.File(filename, 'r')

for key in f.keys():
    print(key) #Names of the groups in HDF5 file.

happy_training = h5py.File('train_happy.h5', "r")
happy_testing  = h5py.File('test_happy.h5', "r")

X_train = np.array(happy_training["train_set_x"][:]) 
y_train = np.array(happy_training["train_set_y"][:]) 

X_test = np.array(happy_testing["test_set_x"][:])
y_test = np.array(happy_testing["test_set_y"][:])

X_train
X_train.shape
y_train
y_train.shape


### DATASET VISUALIZATION
'''
i = random.randint(1,600) # select any random index from 1 to 600
plt.imshow( X_train[i] )
print(y_train[i])

#Let's view more images in a grid format
#Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5
​
#fig, axes = plt.subplots(L_grid, W_grid)
#subplot return the figure object and axes object
#we can use the axes object to plot specific figures at various locations
​
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
​
axes = axes.ravel() #flaten the 15 x 15 matrix into 225 array
​
n_training = len(X_train) #get the length of the training dataset
​
#Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): #create evenly spaces variables 
​
    #Select a random number
    index = np.random.randint(0, n_training)
    #read and display an image with the selected index    
    axes[i].imshow( X_train[index])
    axes[i].set_title(y_train[index], fontsize = 25)
    axes[i].axis('off')
​
plt.subplots_adjust(hspace=0.4)
'''
## Training
#normalize
X_train = X_train/255
X_test = X_test/255

X_train
plt.imshow(X_train[9])
X_train.shape
y_train.shape

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


cnn_model = Sequential()
cnn_model.add(Conv2D(64, 6, 6, input_shape = (64,64,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.2))

cnn_model.add(Conv2D(64, 5, 5, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 128, activation = 'relu'))
cnn_model.add(Dense(output_dim = 1, activation = 'sigmoid'))


cnn_model.compile(loss ='binary_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

epochs = 5
history = cnn_model.fit(X_train,
                        y_train,
                        batch_size = 30,
                        nb_epoch = epochs,
                        verbose = 1)

------------------------------------
### STEP 5: EVALUATING THE MODEL ###
------------------------------------
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

#get the predictions for the test data
predicted_classes = cnn_model.predict_classes(X_test)

predicted_classes.shape
y_test.shape

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Prediction Class = {}\n True Class = {}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)

#axes[i].set_title("Guess{}\n True{}".format(predicted_class[i], y_test[i]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
#Sum the diagonal element to get the total true correct values

from sklearn.metrics import classification_report

print(classification_report(y_test.T, predicted_classes, target_names = target_names))




