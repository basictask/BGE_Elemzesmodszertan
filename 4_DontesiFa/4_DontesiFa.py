# -*- coding: utf-8 -*-

#Graphviz telepítés (mindkét helyről telepíteni kell): 
# https://graphviz.org/download/
# conda install graphviz

import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import preprocessing

import matplotlib.pyplot as plt

import pydotplus # conda install pydotplus

cm = 'viridis' # a diagramok színei globális változó


#%% Adatok ábrázolásához segéd metódus: ponthálóvá alakítás
def get_grid(data):
    x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
    y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))


#%% Modell ábrázolása
def scatter_grid(features, clf_tree):
    xx, yy = get_grid(x_train[features])
    predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    le = preprocessing.LabelEncoder()
    
    for i in range(len(predicted)):
        predicted[i] = le.fit_transform(predicted[i])
    
    predicted = predicted.astype(np.float64)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pcolormesh(xx, yy, predicted, cmap=cm)
    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=le.fit_transform(y_train), s=100, 
                cmap=cm, edgecolors='black', linewidth=1.5);


#%% A döntési fa exportálásához metódus
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, 
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)  
    graph.write_png(png_file_to_save)
    
    
#%% Adatok beolvasása
df = pd.read_csv('halak.csv', header=0, sep=';', encoding='ISO-8859-2')

X = df[['Hossz1', 'Magassag']] # Független változók
Y = df['Faj'] # Célváltozó

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

le = preprocessing.LabelEncoder() # Címkekódoló a fajták számmal való reprezentálásához

plt.figure(figsize=(8,8))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=le.fit_transform(Y), cmap=cm)
plt.show()


#%% Döntési fa modell tanítása
clf_tree_2var = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

clf_tree_2var.fit(x_train, y_train)

scatter_grid(['Hossz1','Magassag'], clf_tree_2var)


#%% Kép generálása a modellből és mentés .png-ként
tree_graph_to_png(tree=clf_tree_2var, feature_names=['Hossz1', 'Magassag'],
                  png_file_to_save='Hossz_Magassag_dontes.png')


#%% Predikció
y_pred = clf_tree_2var.predict(x_val)

# Egy egyszerű DataFrame a célváltozó értékeinek megtekintésére az egyes kísérletekben
df_pred = pd.DataFrame({'original': y_val, 
                       'predicted': y_pred})

df_pred['match'] = [1 if x==y else 0 for x,y in zip(df_pred['original'], 
                                                    df_pred['predicted'])]

print(accuracy_score(y_val, y_pred))


#%% Mégegy próba, több változóval
varlst = ['Suly', 'Hossz1', 'Hossz2', 'Hossz3', 'Magassag', 'Szelesseg']

X = df[varlst] # Független változók
Y = df['Faj'] # Célváltozó

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

clf_tree_multivar = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

clf_tree_multivar.fit(x_train, y_train)

y_pred_tree = clf_tree_multivar.predict(x_val)

df_pred['tree_pred'] = y_pred_tree

df_pred['tree_match'] = [1 if x==y else 0 for x,y in zip(df_pred['original'], 
                                                         df_pred['tree_pred'])]

print(accuracy_score(y_val, y_pred_tree))

tree_graph_to_png(tree=clf_tree_multivar, feature_names=varlst,
                  png_file_to_save='Multivar_dontes.png')
