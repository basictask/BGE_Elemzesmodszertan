# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

# Együttes tanulás
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.stats import mode

cm='viridis'

# Gradiens turbózás
import lightgbm as lgb
import seaborn as sns
import time
import pydotplus

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz

from matplotlib.animation import FuncAnimation


#%% Döntési határok felrajzolására függvény
def plot_decision_boundary(clf, X, y, axes=[0, 50, 0, 20], alpha=0.5, contour=True):
    X = np.array(X)
    y = np.array(y)
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    le = preprocessing.LabelEncoder()
    for i in range(len(y_pred)):
        y_pred[i] = le.fit_transform(y_pred[i])
    
    y_pred = y_pred.astype(np.float64)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])    
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
    
#%% Adatok beolvasása
df = pd.read_csv('halak.csv', header=0, sep=';', encoding='ISO-8859-2')

X = df[['Hossz1', 'Magassag']] # Független változók
Y = df['Faj'] # Célváltozó

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

le = preprocessing.LabelEncoder() # Címkekódoló a fajták számmal való reprezentálásához

plt.figure(figsize=(8,8))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=le.fit_transform(Y), cmap=cm)
plt.show()

df_pred = pd.DataFrame({'original': y_val})


#%% Kétváltozós döntési fa tanítása
clf_tree_2var = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

clf_tree_2var.fit(x_train, y_train)

df_pred['2var_tree'] = clf_tree_2var.predict(x_val) # Predikciók készítése az adathalmazra


#%% Bagging
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), # Milyen modellel akarjuk véghez vinni a Bagging-et
                            n_estimators=500, # Hány modellt tanítson az algoritmus
                            max_samples=120,  # Mennyi mintaegyed különüljön el az egyes modellek tanításának
                            bootstrap=True, 
                            random_state=42)

bag_clf.fit(x_train[['Hossz1','Magassag']], y_train) # tanítás

y_pred_bag = bag_clf.predict(x_val[['Hossz1','Magassag']]) # predikció

df_pred['Bagging'] = y_pred_bag # ellenőrzés
df_pred['Bag_match'] = [1 if x==y else 0 for x,y in zip(df_pred['original'], 
                                                        df_pred['Bagging'])]
print(accuracy_score(y_val, y_pred_bag))
    
    
#%% Döntési fa vs. Bagging ábrázolása
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(clf_tree_2var, X[['Hossz1', 'Magassag']], Y)
plt.title("Döntési Fa", fontsize=14)

plt.sca(axes[1])
plot_decision_boundary(bag_clf, X[['Hossz1', 'Magassag']], Y)
plt.title("Döntési fa + Bagging", fontsize=14)
plt.ylabel("")
plt.show()


#%% Legjobb döntési fa megtalálása GridSearch-el
x_train = np.array(x_train) # adathalmazok átalakítása np objektumokká
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

params = {'max_leaf_nodes': list(range(2, 100)), # melyik regularizációs hiperparaméterek kombinációjában kell keresni?
          'min_samples_split': [2, 3, 4],
          'min_samples_leaf': [1, 2, 3]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=1, cv=3) # hiperparam. tuning

grid_search_cv.fit(x_train, y_train) # hiperparaméter tuning illesztés

best = grid_search_cv.best_estimator_ # legjobb prediktor megtalálása az együttesben


#%% A tanító adathalmaz 1000 random kiválasztott részhalmaza
n_trees = 1000
n_instances = 100 

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(x_train) - n_instances, random_state=42) # véletlen szétdarabolás

for mini_train_index, mini_test_index in rs.split(x_train): # a szétdarabolt részhalmazok listába illesztése
    X_mini_train = x_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
    
    
#%% A legjobb modell klónozása és minden modell segítségével predikció készítés
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)] # Legjobb modell erdővé klónozása

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets): # Az erdő minden fájával predikció készítése
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(x_val)
    accuracy_scores.append(accuracy_score(y_val, y_pred))  # Pontosság mérése

np.mean(accuracy_scores) # Átlagos predikció pontosság


#%% Együttes tanulókkal predikció készítése
y_pred_ensemble = np.empty([n_trees, len(x_val)], dtype='O') # Az együttes predikcióinak adatstruktúra

for tree_index, tree in enumerate(forest): 
    y_pred_ensemble[tree_index] = tree.predict(x_val) # Predikciók készítése
    
y_pred_majority_votes, n_votes = mode(y_pred_ensemble, axis=0) # A leggyakoribb predikció legyen a végleges

ensemble_preds = pd.Series(y_pred_majority_votes[0], index=df_pred.index) # Series-é alakítás

df_pred['Ensemble'] = ensemble_preds # Ellenőrző adathalmazra ráfűzés

df_pred['Ensemble_match'] = [1 if x==y else 0 for x,y in zip(df_pred['original'], 
                                                             df_pred['Ensemble'])]

print(accuracy_score(y_val, ensemble_preds))


#%%###########################################################################
# GBDT modellek sejtek osztályozására: 0:Nem Rákos, 1: Rákos
# Adatszerkezet felállítása
df = pd.DataFrame(data = load_breast_cancer().data, columns=load_breast_cancer().feature_names)
df['y'] = list(load_breast_cancer().target) # Célváltozó
df.head()


#%% Korrelációs mátrix heatmap-pel
corr = df.corr() 

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(22,18)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


#%% Gyenge változók kidobálása
df = df.drop(list(corr['y'][abs(corr['y'])<0.5].index),axis=1) # Aminek kevesebb mint közepesen erős a korrelációja


#%% Tanítás-teszt szeparáció 8:2 arányban
mask = np.random.randn(len(df)) < 0.8 # tesztmaszk
x_train = df[mask].iloc[:,:15] 
x_test = df[~mask].iloc[:,:15]
y_train = df[mask].iloc[:,15:]
y_test = df[~mask].iloc[:,15:]


#%% Paraméterek könyvtára
params = {
    'boosting_type': 'gbdt', # Gradient Boosting Decision Tree
    'objective': 'binary', # Logiszikus
    'num_leaves': 31, # Levelek száma
    'min_data_in_leaf': 20, # Terminális régiókba bekerülő adatmennyiség
    'max_depth': 10, # Elérhető maximális mélység
    'max_bin': 255, # Legtöbb kosár amibe a változó kerülhet
    'learning_rate': 0.1, # Tanulási sebesség
    'metric': [11,12], # Ehhez nem nyúlunk
    'bagging_fraction': 0.8, # Újramintázási mennyiség
    'bagging_freq': 5 # Újramintázási gyakoriság
}


#%% Függvény egy trbózó tanítására és predikciók lekérésére
def experiment(objective, label_type, x_train, x_test, y_train, y_test, i):
    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False) # Train LGBM Datasetté alakítása
    lgb_test = lgb.Dataset(x_test, y_test, free_raw_data=False) # Test LGBM Datasetté alakítása
    
    params['objective'] = objective # Célfüggvény
    
    if(i==0):
        gbm = lgb.train(params, # LGBM modell létrehozása
                        lgb_train, 
                        valid_sets=[lgb_train, lgb_test], # Validációs adathalmazok
                        num_boost_round=10) # Hány körös legyen a turbózás
        gbm.save_model(str(objective)+'gbmodel.txt') # Modell mentése
    else:
        gbmprev = lgb.Booster(model_file=str(objective)+'gbmodel.txt') # Előző modell betöltése ha melegindítás
        gbm = lgb.train(params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        num_boost_round=10,
                        init_model=gbmprev) # Kezdeti modell megadása, amit tovább kell építenie
        gbm.save_model(str(objective)+'gbmodel.txt') 
      
    y_fitted = pd.Series(gbm.predict(x_test, num_iteration=gbm.best_iteration)) # Predikciók lekérése    
    y_pred = pd.DataFrame({'y_pred': y_fitted}) # Df-é alakítás
     
    y_true = y_test.copy().set_index(np.arange(len(y_test)))
    
    return pd.concat([y_true, y_pred], axis=1)


#%% Viselkedés vizsgálata bináris és szabadon választott célponttal
# Választható: regression, regression_l1, huber, fair, possion, quantile, mape, gamma, tweedie, binary
#              multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg


A = [experiment('binary', 'binary', x_train, x_test, y_train, y_test, k) for k in range(10)] # GBDT bináris céllal

second_target = 'tweedie'
tree_num = 10
B = [experiment(second_target, 'binary', x_train, x_test, y_train, y_test, k) for k in range(tree_num)] # Másodlagos céllal


#%% Pontosság mérése adott iterációk alatt
accuracy_binary = lambda D: [accuracy_score(x['y'], x['y_pred'].round()) for x in D] # Adatpontok összerendezése
accuracy_mean = lambda D: [mean_squared_error(x['y'], x['y_pred']) for x in D]

def accplot(accdata, objective): # Pontosságot mérő függvény matplotlib fekete mágiával
    fig = plt.figure(figsize=(10,8))
    ax2 = fig.add_subplot(2,1,2)
    ax1 = fig.add_subplot(2,1,1, sharex=ax2) 
    fig.suptitle(objective + ' classification', fontsize=16)
    ax1.plot(accuracy_binary(accdata), '-ob')
    ax1.set_title('Bináris pontosság')
    ax2.plot(accuracy_mean(accdata))
    ax2.set_title('Valószínűségi rezidum')
    plt.xticks(np.arange(0,len(accdata)))
    plt.show()

accplot(A, 'binary') # Pontosság elsődleges
accplot(B, second_target) # És másodlagos célpontra


#%% Kontingencia tábla, hőtérkép
y_predfinal = pd.DataFrame({'pred': A[len(A)-1]['y_pred'].copy().round()})
cmx = confusion_matrix(y_test, y_predfinal) # Ezzel hozza létre. 2 célosztály=2*2 tábla

print(classification_report(y_test, y_predfinal))

plt.figure(figsize=(5,5))
confusion_matrix_df=pd.DataFrame(cmx,('No cancer', 'Cancer'),('No cancer', 'Cancer'))

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d") # Korrelációs mátrix hőtérképpé
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14) # Tengelycímkék
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize = 14)

plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
plt.show()


#%% Elemek rendezése
ordergen = lambda D: [x.sort_values(by=['y_pred']).reset_index().drop('index',axis=1) for x in D.copy()]


#%% Függvény animálása elsődleges célponttal
%matplotlib qt
C = ordergen(A)

fig, ax = plt.subplots(figsize=(15,10)) # Animálandó diagram felvétele
ax.set_xlim(0,len(y_test))
ax.set_ylim(-0.1, 1.1)
ax.set_title('Célpont: bináris')
line, = ax.plot(0,0) # Kirajzolandó vonal létrehozása

x_data = []
y_data = []

def animation_frame(i): # Egyetlen képkocka kirajzolása
    line.set_xdata(np.arange(0,len(C[i]))) # Vonal inícializálása
    fig.suptitle('GBDT Logisztikus regresszió (' + str(i+1)+' fa)', fontsize=20) # Ez a cím miatt kell
    y_data = []
    line.set_ydata(C[i]['y_pred']) # Vonal szerkesztése
    ax.plot(C[0]['y'],'o', color='black') # Diagramra kirajzolás
    return line,

animation = FuncAnimation(fig, func=animation_frame, frames=range(10), interval=200) # Animálás
plt.show()


#%% Függvény animálás másodlagos célpontra
C = ordergen(B)

fig, ax = plt.subplots(figsize=(15,10))
ax.set_xlim(0,len(y_test))
ax.set_ylim(-0.1, 1.1)
ax.set_title('Célpont: ' + second_target)
line, = ax.plot(0,0)

x_data = []
y_data = []

animation = FuncAnimation(fig, func=animation_frame, frames=range(len(C)), interval=200)
plt.show()


#%% Egy teljes döntési fa létrehozása az adathalmazra
# Létrehoz egy teljes döntési fát
decisionTree_unpruned = DecisionTreeClassifier()
decisionTree = DecisionTreeClassifier(max_depth=5)

# Optimalizálni az adathalmazra
decisionTree_unpruned = decisionTree_unpruned.fit(X=x_train, y=y_train)
decisionTree = decisionTree.fit(X=x_train, y=y_train)

tree_str = export_graphviz(decisionTree, filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(tree_str)  
graph.write_png('Cancer_Dectree.png')
