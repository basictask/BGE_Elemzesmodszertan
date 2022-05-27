# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

#%%
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau


#%%
import keras
from tensorflow.keras.models import load_model
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import backend as K


#%%2 Adat betöltése
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop("label", axis=1) 

# Normalizálás
X_train = X_train / 255.0
test = test / 255.0

# Átformálás
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Vektorkódolás
Y_train = to_categorical(Y_train, num_classes = 10)

# Training és validáció szétválasztása
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


#%% Modell felállítása
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                 activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="linear", name='preds'))


# Optimalizáló
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Tanulási sebesség csökkentő
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)


#%%
epochs = 5
batch_size = 128

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
                    validation_data=(X_val, Y_val), verbose=1)

model.summary()

model.save('digit_model_linear.h5')

model = load_model('digit_model_linear.h5')           
           

#%% Modell betöltése
model = load_model('digit_model_linear.h5')


#%% Szaliencia vizualizáció
import vis
from vis.visualization import visualize_saliency
from vis.utils import utils
from tensorflow.keras import activations

for target in range(10):
    class_idx = target
    indices = np.where(Y_val[:, class_idx] == 1.)[0]
    idx = indices[0]
        
    layer_idx = utils.find_layer_idx(model, 'preds')
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=X_val[idx])
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(X_val[idx][..., 0])
    ax[0] = plt.plot(X_val[idx][..., 0])
    ax[1].imshow(grads, cmap='jet')
    plt.show()


#%%
class_idx = 0
indices = np.where(Y_val[:, class_idx] == 1.)[0]
idx = indices[0]    
    
grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                           seed_input=X_val[idx], backprop_modifier='guided')
