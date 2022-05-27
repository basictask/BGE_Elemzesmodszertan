# -*- coding: utf-8 -*-
#1 Set up packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns

from sklearn.model_selection import train_test_split # 0.23.2
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf # 2.4.0
from tensorflow.keras.models import load_model
from keras.utils.np_utils import to_categorical # One-hot kódolás
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

from keract import get_activations, display_activations
from keract.keract import display_heatmaps

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.disable_eager_execution()
sns.set(style='white', context='notebook', palette='deep')


#%%2 Adatok betöltése
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop("label", axis=1) 

g = sns.countplot(Y_train)
Y_train.value_counts()


#%%3 Számok mutatása egy táblázatban
plt.figure(figsize=(14,12))
for digit_num in range(0,70):
    plt.subplot(7,10,digit_num+1)
    grid_data = test.iloc[digit_num].to_numpy().reshape(28,28)
    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()


#%%4 Null és hiányzó értékek keresése
print(X_train.isnull().any().describe())
print('---------------')
print(test.isnull().any().describe())


#%%5 Előfeldolgozás
# Normalizálás
X_train = X_train / 255.0
test = test / 255.0

# Átalakítás 28x28x1 -es mátrixokká
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Vektorkódolás
Y_train = to_categorical(Y_train, num_classes = 10)


#%%6 Train-Test szétválasztás
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

g = plt.imshow(X_train[0][:,:,0])


#%%7 Aktivációs függvények vizualizálása
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

z = np.linspace(-5, 5, 200)
plt.figure(figsize=(11,4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Aktivációs függvények", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=1, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
#plt.legend(loc="center right", fontsize=14)
plt.title("Deriváltjaik", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

plt.show()


#%%8 Szeparációs függvények vizualizálása
def mlp_xor(x1, x2, activation):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)

x1s = np.linspace(-0.2, 1.2, 100)
x2s = np.linspace(-0.2, 1.2, 100)
x1, x2 = np.meshgrid(x1s, x2s)

z1 = mlp_xor(x1, x2, activation=relu)
z2 = mlp_xor(x1, x2, activation=sigmoid)

plt.figure(figsize=(10,4))

plt.subplot(121)
plt.contourf(x1, x2, z1)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "y^", markersize=20)
plt.title("Activation function: relu", fontsize=14)
plt.grid(True)

plt.subplot(122)
plt.contourf(x1, x2, z2)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "y^", markersize=20)
plt.title("Activation function: sigmoid", fontsize=14)
plt.grid(True)


#%%9 CNN Modell felépítése
## Input -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Output
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

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
model.add(Dense(10, activation="softmax", name='preds')) # Eloszlási valószínűség minden osztályra

model.summary()


#%%10 Modell kirajzolása
tf.keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)


#%%11 Otimalizáló és teljesítmény metrika
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Tanulási sebesség dinamikus csökkentése
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)


#%%12 Rétegek vizualizálása
keract_inputs = X_train[3:4] # ez egy 3-as szám
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=False)


#%%13 Heatmap-ek vizualizálása
keract_inputs = X_train[3:4]
activations = get_activations(model, keract_inputs)
display_heatmaps(activations, keract_inputs[0], fix=False)


#%%14 Gyorsított tanítási eljárás
history = model.fit(X_train[:2000], Y_train[:2000], batch_size=1000, epochs=5, 
                    validation_data=(X_val[:500], Y_val[:500]), verbose=1)


#%%18 Modell kiértékelése
def measure_model(hist):
    plt.figure(figsize=(9,9))
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    
    # Train és teszt veszteségek 
    plt.figure(figsize=(9,9))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

measure_model(history)


#%%15 Adataugmentálás
datagen = ImageDataGenerator(featurewise_center=False, 
                             samplewise_center=False,
                             featurewise_std_normalization=False, 
                             samplewise_std_normalization=False, 
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False)
datagen.fit(X_train)


#%%16 Modell taníttatása
history = model.fit_generator(datagen.flow(X_train, Y_train), 
                              epochs=30,
                              validation_data=(X_val, Y_val), 
                              verbose=1, 
                              callbacks=[learning_rate_reduction])

# measure_model(history) # Ha valaki lefuttatja ezt, mérje meg a teljesítményt


#%%17 Modell betöltése / elmentése
model = load_model('digit_model.h5')

# model.save('digit_model.h5') 


#%%19 Konfúziós mátrix kirajzolása
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    

Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10)) 


#%%20 Normalizált konfúziós mátrix
row_sums = confusion_mtx.sum(axis=1, keepdims=True)
norm_conf_mx = confusion_mtx / row_sums

np.fill_diagonal(norm_conf_mx, 0)

plt.figure(figsize=(6,6))
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.ylabel('actual classes')
plt.xlabel('predicted classes')
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.show()


#%%21 Csak 4 és 9 számjegyek kirajzolása
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


def revert(data):
    final = []
    for x in data: 
        for i, y in zip(np.arange(10), x):
            if(y==1):
                final.append(i)
    return pd.Series(final)

Y_train_pred = model.predict(X_train)

Y_train_reverted = revert(Y_train)
Y_train_pred_reverted = revert(Y_train_pred)

cl_a, cl_b = 4, 9
X_aa = X_train[(Y_train_reverted==cl_a) & (Y_train_pred_reverted==cl_a)]
X_ab = X_train[(Y_train_reverted==cl_a) & (Y_train_pred_reverted==cl_b)]
X_ba = X_train[(Y_train_reverted==cl_b) & (Y_train_pred_reverted==cl_a)]
X_bb = X_train[(Y_train_reverted==cl_b) & (Y_train_pred_reverted==cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()



#%%22 A hibás kimenetek megjelenítése
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(2, 3, sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],
                                                                               obs_errors[error]))
            n += 1

# Rossz predikciók valószínűsége
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Becsült valószínűsége az igazi értékeknek a hibás adathalmazon 
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# A becsült és valós értékek közötti különbség
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Rendezett lista a különbségekről
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 hibás adat
most_important_errors = sorted_dela_errors[-6:]

# Top 6 hibás adat kirajzolása
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


#%%23 Predikció
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results, name="Label")


#%%24 Saját képen predikció
img = image.imread('black_28x28.png')

data = pd.DataFrame([list(img[i,:,0]) for i in range(28)])

temp = test[0:1]

temp2 = img[:,:,0:1]

temp[0] = temp2

results = model.predict(temp)

results = results.T

pred = -1
for i,j in zip(range(len(results)), results):
    if(j==max(results)):
        pred=i

plt.title('A becsult ertek: '+str(pred), fontsize=20)
plt.imshow(img)
