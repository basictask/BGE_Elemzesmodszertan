# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM


#%% Az Olivetti arcok osztályozása GMM-mel
olivetti = fetch_olivetti_faces()
print(olivetti.DESCR)
print(olivetti.target)


#%% Train-test szétválasztás
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]


#%% Főkomponenselemzéssel egyszerűsítsük
pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

pca.n_components_


#%% Gauss-keverék tanítása
gm = GMM(n_components=40, random_state=42)
y_pred = gm.fit_predict(X_train_pca)


#%% Generáljunk új arcokat véletlen mintavétellel!
n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)


#%% Ábárzoljuk ezeket
def plot_faces(faces, labels, n_cols=5, title=""):
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    plt.title(title)
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face.reshape(64, 64), cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

plot_faces(gen_faces, y_gen_faces)


#%% Módosítsunk néhány arcot, majd nézzük meg, a modell felismeri-e anomáliaként
n_rotated = 4 # Forgasson el 4 képet
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3 # Fordítson fejjel lefele 3 képet
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3 # Sötétítsen el 3 képet
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
darkened = darkened.reshape(-1, 64*64)
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened] # Összefűzés 1 adathalmazba
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces, y_bad)


#%% Anomália detekció
X_bad_faces_pca = pca.transform(X_bad_faces)

print('Anomáliák valószínűségei')
print(gm.score_samples(X_bad_faces_pca))
print()
print('Eredeti arcok valószínűségei')
print(gm.score_samples(X_train_pca[:10]))


#%% Új adatok generálása az MNIST adathalmaz alapján #########################
digits = load_digits()
digits.data.shape


#%% Ábrázoljuk az adatokat
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
plot_digits(digits.data)


#%% Főkomponenselemzés az adatokon
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
print(data.shape)


#%% Komponensek számának megtalálása
n_components = np.arange(50, 210, 10)
models = [GMM(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics);


#%% Illesszünk Gaussi keveréket az adathalmazra
gmm = GMM(140, covariance_type='full', random_state=0)
gmm.fit(data)
print('konvergált: ', gmm.converged_)


#%% Vegyünk mintát az adatokból, ezzel új adatokat csinálva
data_new = gmm.sample(100)[0]
print(data_new.shape)


#%% Próbáljuk meg visszaalakítani
digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
