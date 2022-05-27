# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#%% Random adatok
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)


#%% Ridge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


reg = Ridge(alpha=0.1, solver="cholesky", random_state=42)

model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("std_scaler", StandardScaler()),
    ("regul_reg", reg)])

model.fit(X, y)
y_new_regul = model.predict(X_new)
plt.plot(X_new, y_new_regul)
plt.plot(X, y, "b.", linewidth=3)


#%% Lasso
from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.4)

model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=4, include_bias=False)),
    ("regul_reg", reg)])

model.fit(X, y)

y_new_regul = model.predict(X_new)
plt.plot(X_new, y_new_regul)
plt.plot(X, y, "b.", linewidth=3)


#%% Early stopping
from sklearn.model_selection import train_test_split
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)


#%% Egy egyszerű Early Stopping implementáció
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

poly_scaler = Pipeline([ # Kis Feature Engineering
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, # A modell: ez lehet bármi aminek kiszámolható a hibája
                       tol=-np.infty, # Megállítási kritérium
                       warm_start=True, # Megtartja az előző modellt, és azt tanítja tovább
                       penalty=None, # Bünti: Próbáljuk meg regularizálni!
                       learning_rate="constant", # Van-e adaptív tanulási sebesség
                       eta0=0.0005, # Kezdeti tanulási sebesség
                       random_state=42) # Véletlenszám-generáló bázisértéke

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # ott folytatja ahol abbahagyta
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    
    if val_error < minimum_val_error: # Early Stopping kriterium
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)


#%% Elasztikus Háló
import pandas as pd
df = pd.read_csv('halak.csv', header=0, sep=';', encoding='ISO-8859-2')

df = df[['Suly', 'Hossz1', 'Hossz2', 'Hossz3', 'Magassag', 'Szelesseg']]

df_X = df[['Hossz1']]

# df_X = df[['Magassag']]

poly_scaler = Pipeline([("std_scaler", StandardScaler())])

X = poly_scaler.fit_transform(df_X)
Y = df['Hossz1']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, random_state=420)


#%% Ridge vs, Lasso 
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model

errorlog = {}
for i in range(1, 10, 1):
    alpha = i/10
    model = ElasticNetCV(l1_ratio=alpha, fit_intercept=True, normalize=True)
    
    model.fit(X_train, y_train)
    
    y_pred =  model.predict(np.array(X_val).reshape(-1,1))
    
    errorlog[alpha] = mean_squared_error(y_val, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.scatter(X_val[:,0], y_pred, alpha=0.8, label='predicted')
    ax.scatter(X_val[:,0], y_val, alpha=0.8, label='observed')
    
    ax.legend()
    plt.show()


#%% Multitask Elastic Net
from sklearn.linear_model import MultiTaskElasticNetCV # Többváltozós elasztikus háló

df_X = df[['Hossz1', 'Hossz2', 'Hossz3', 'Magassag', 'Szelesseg']]

poly_scaler = Pipeline([("std_scaler", StandardScaler())]) # X átalakítója 

X = poly_scaler.fit_transform(df_X) # X sztenderdizálása

Y = np.array(df['Hossz1']).reshape(-1,1) # Y definiálás

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, random_state=420) # train-test kettécsapás


errorlog_multi = {}
for i in range(1, 10, 1):
    alpha = i/10
    model = MultiTaskElasticNetCV(l1_ratio=alpha, fit_intercept=True)
    
    model.fit(X_train, y_train)
    
    y_pred =  model.predict(X_val)
    
    errorlog_multi[alpha] = mean_squared_error(y_val, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.scatter(X_val[:,0], y_pred, alpha=0.8, label='predicted')
    ax.scatter(X_val[:,0], y_val, alpha=0.8, label='observed')
    
    ax.legend()
    plt.show()
