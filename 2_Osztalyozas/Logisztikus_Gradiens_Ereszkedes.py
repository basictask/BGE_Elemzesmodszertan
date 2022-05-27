#%% # -*- coding: utf-8 -*-
# Könyvtárak
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 1000}) # Ezzel óvatosan

#%% Adatok generálása
X = np.array(np.linspace(0, 10, 100))

y = []
for i in range(len(X)):
    if (X[i] < 5):
        y.append(0)
    else:
        y.append(1)
        
y = np.array(y)


#%% Adatok ábrázolása 
plt.figure(figsize=(12, 7))
plt.scatter(X, y)


#%% Együtthatók kezdeti értéke
b0 = 0
b1 = 0
alpha = 0.001


#%% Függvény definiálása
def sigmoid(b0, b1, x):
    return 1/(1 + np.exp((-1)*(b0 + b1*x)))

P = sigmoid(b0, b1, X)


#%% Függvény Ábrázolása
plt.figure(figsize=(12, 7))
plt.scatter(X, y)
plt.plot(X, P, color='red')


#%% Kezdeti Hiba értéke
np.sum(-y*np.log(P) - (1-y)*np.log(1-P))


#%% Értékek közelítése 1 iterációval
#együtthatók újraszámolása
b0 = b0 - np.sum(P - y)*alpha
b1 = b1 - np.sum((P - y)*X)*alpha

#sigmoid függvény újraszámolása
P = sigmoid(b0, b1, X)

def plot_log():
    plt.figure(figsize=(12, 7))                         #méret
    plt.scatter(X, y)                                   #adat pontok
    plt.plot(X, P, color='red')                         #sigmoid függvény
    plt.plot(X, X*0+0.5, color='grey', linestyle='--')  #szürke szaggatott vonal p=0,5-höz

plot_log()


#%% Értékek közelítése 1000 iterációval
for i in range(1000):
    b0 = b0 - np.sum(P - y)*alpha
    b1 = b1 - np.sum((P - y)*X)*alpha
    P = 1/(1 + np.exp((-1)*(b0 + b1*X)))
    plot_log()
    
print('összes hiba nagysága: ', np.sum(-y*np.log(P) - (1-y)*np.log(1-P)))
