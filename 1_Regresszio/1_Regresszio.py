#%% Lineáris Regresszió 
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


#%% Adatok beolvasása
df = pd.read_csv('gdp_kisajatitas.csv')
df.head()


#%% Adatok ábrázolása
plt.style.use('seaborn')

df.plot(x='avexpr', y='logpgp95', kind='scatter')
plt.show()


#%% Adatok tisztítása
df_clean = df.dropna()
df_clean['const'] = 1 # az y tengelymetszet nem lehet 0, mert nincs 0 GDP ország

# Változók felvétele
x = df_clean.loc[:, 'avexpr']
y = df_clean.loc[:, 'logpgp95']
labels = df_clean.loc[:, 'shortnam']


#%% Rergressziós modell felállítása
reg1 = sm.OLS(endog=y, exog=df_clean[['const', 'avexpr']]) #endog: függő változó, exog: független változó

type(reg1)

# A tanítás eljárása
results = reg1.fit()

type(results)

# A modell
print(results.summary())


#%% Lineáris trendvonal illesztése
fig, ax = plt.subplots()
ax.scatter(x, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (x.iloc[i], y.iloc[i]))
    
line = np.poly1d(np.polyfit(x, y, 1))(np.unique(x))
ax.plot(np.unique(x), line, color='black')

plt.show()


#%% Predikciók készítése
fig, ax = plt.subplots()

ax.scatter(df_clean['avexpr'], results.predict(), alpha=0.8, label='predicted')

ax.scatter(df_clean['avexpr'], df_clean['logpgp95'], alpha=0.8, label='observed')

ax.legend()
plt.show()


#%% Gradiens ereszkedés
# Paraméterek felvétele
b0 = 0
b1 = 0

#Egyenes egyenlete
f = b0 + b1*x

#Egyenes ábrázolása
def plotf():
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y)
    plt.plot(x, f, color='red')

plotf()

#Eltérés-négyzet
np.sum((f - y)**2)


#%% b0 és b1 közelítése az optimálishoz
#Tanulási sebesség
alpha = 0.00001

b0 = b0 - np.sum(2*(b0 + b1*x - y))*alpha
b1 = b1 - np.sum(2*(b0 + b1*x - y)*x)*alpha

print('b0: ', b0, ' b1: ', b1)


# Regresszió fgv újraszámolása
f = b0 + b1*x

plotf()


#%% Hiba számolása ciklikusan
b0=0
b1=0

hibakovetes = []
alpha = 0.00001

for i in range(1000):
    b0 = b0 - np.sum(2*(b0 + b1*x - y))*alpha
    b1 = b1 - np.sum(2*(b0 + b1*x - y)*x)*alpha
    hibakovetes.append(np.sum((b0 + b1*x - y)**2))
    
    
f = b0 + b1*x
print('b0: ', b0, ' b1: ', b1)
print('hiba négyzetösszeg: ', np.sum((f - y)**2))
plotf()

#%%
plt.figure(figsize=(12, 7))
plt.scatter(np.linspace(0, len(hibakovetes), len(hibakovetes)), hibakovetes)
