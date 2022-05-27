#%% Import libs
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from seaborn import scatterplot as scatter
import math
from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters()
from fcmeans import FCM


#%% Read Data
termek_tulajdonsag = pd.read_csv('termek_tulajdonsag.csv', header=0, sep=',')

df_termek = termek_tulajdonsag.set_index('z_soid')

df_termek.loc[df_termek['haszonkulcs']>200, 'haszonkulcs'] = 200


#%% Get prices data
arak = pd.DataFrame(df_termek['bevetel'] / df_termek['vszam'], columns=['ar']).round()

def choose_boost(n):
    if(n<29990):
        return 0
    elif(n>=29990 and n<51077):
        return 1
    elif(n>=51077 and n<83330):
        return 2
    else:
        return 3
        
arak['cat'] = [choose_boost(n) for n in arak['ar']]

#df_termek = pd.concat([df_termek, arak['ar']], axis=1)

scaled_termek = preprocessing.scale(df_termek)


#%% Fuzzy C-means
fcm = FCM(n_clusters=5, random_state=42, max_iter=1000)
fcm.fit(scaled_termek)

fcm_centers = fcm.centers
u = pd.DataFrame(fcm.u)
fcm_labels  = fcm.u.argmax(axis=1)

y_pred_fuzzy = fcm.predict(scaled_termek)


#%% Optimal number of clusters
from sklearn.cluster import KMeans
kmeans_per_k = [KMeans(n_clusters=k, init='k-means++', n_init=1, 
                       random_state=42).fit(scaled_termek) for k in range(1,10)]

inertia = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8,8))
plt.plot(range(1,10), inertia, 'bo-')
plt.xlabel('$k$', fontsize=14)
plt.ylabel('inertia', fontsize=14)
plt.show()


#%% Silhouette score
from sklearn.metrics import silhouette_score

silhouettes = [silhouette_score(scaled_termek, model.labels_) for model in kmeans_per_k[1:]]

plt.figure(figsize=(8,4))
plt.plot(range(2,10), silhouettes, 'bo-')
plt.xlabel('$k$', fontsize=14)
plt.ylabel('silhouette_score', fontsize=14)
plt.show()


#%% K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=100, max_iter=1000, random_state=42)
kmeans.fit(scaled_termek)

print('inertia:', kmeans.inertia_)

y_pred = kmeans.predict(scaled_termek)


#%% PCA
from sklearn.decomposition import PCA

var_ratio = 2
pca = PCA(n_components=var_ratio)
pca.fit(scaled_termek)
pca_data = pca.transform(scaled_termek)

print('components with 95% preserved variance: ', pca.n_components_)
print('Sum of explained variance ratio: ', np.sum(pca.explained_variance_ratio_))


#%% Visualize PCA
fs=20

create_pca_frame = lambda pcd: pd.DataFrame(pcd, columns = [str(x) for x in range(pcd.shape[1])])

pca_frame = create_pca_frame(pca_data)

# pca_frame['pred'] = pd.Series(y_pred) #Klaszterek 

pca_frame['pred'] = pd.Series(y_pred_fuzzy) #Fuzzy C-means klaszterek

# pca_frame['pred'] = arak.reset_index()['cat'] #Árkategóriák

# Ha ezt kikommentelve akarjuk futtatni, a lenti kodokat indentalni kell!

# scaled_frame = create_pca_frame(scaled_termek)
# for x,y in zip(range(4), ['vszam','bevetel','haszonkulcs','vnap']):    
#     pca_frame['pred'] = scaled_frame[str(x)]

#cluster center
centers = kmeans.cluster_centers_
pca_centers = pd.DataFrame(pca.transform(centers), columns=['0','1'])

plt.figure(figsize=(12,12))
plt.title('K-közép algoritmus és főkomponenselemzés', size=fs+5)
# cmap = sns.color_palette("bright")[:5]
ax = scatter(x='0', y='1', hue='pred', data=pca_frame)
ax = scatter(x='0', y='1', color=".2", marker="+", data=pca_centers)
ax.legend(loc='upper right', fontsize=fs)
ax.tick_params(labelsize=fs)
ax.set_xlabel('Főkomponens1', size=fs)
ax.set_ylabel('Főkomponens2', size=fs)
plt.show()


#%% Silhouette diagrams
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl

plt.figure(figsize=(15,12))

for k in (3,4,5,6):
    plt.subplot(2, 2, k-2)
    y_pred = kmeans_per_k[k-1].labels_
    silhouette_coeffs = silhouette_samples(pca_data, y_pred)
    
    padding = len(pca_data) // 30
    
    pos = padding
    ticks = []

    for i in range(k):
        coeffs = silhouette_coeffs[y_pred == i]
        coeffs.sort()
        
        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos+len(coeffs)), 0, coeffs, facecolor=color, 
                          edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos+=len(coeffs) + padding
        
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    
    if k in (3,5):
        plt.ylabel('cluster')
    
    if k in (5,6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel('Silhouette Coefficient')
    else:
        plt.tick_params(labelbottom=False)
        
    plt.axvline(x=silhouettes[k-2], color="red", linestyle='--')
    plt.title('$k={}$'.format(k), fontsize=16)

plt.suptitle('Termékek klaszterszámának sziluett diagramjai', fontsize=16)
plt.show()


#%% Loading Scores
loading_scores = pd.Series(pca.components_[0])
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

loadings = pd.Series(loading_scores)
factors = pd.Series(df_termek.columns)
print(pd.concat([loadings, factors], axis=1))


#%% Scree Plot
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['FK' + str(x) for x in range(1, len(per_var)+1)]
plt.figure(figsize=(8,8))
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Magyarázóerő (%)')
plt.xlabel('Főkomponens')
plt.title('Scree Plot')
plt.show()


#%% Get cluster descriptions
df_cluster = df_termek.copy().reset_index()

df_cluster['pred'] = pd.Series(y_pred)
df_cluster['cat'] = pd.Series(arak.reset_index()['cat'])

n = kmeans.n_clusters

df_stat = pd.DataFrame()
for i in range(n):
    stat = df_cluster[df_cluster['pred']==i][['vszam','bevetel','haszonkulcs','vnap']].describe()    
    df_stat = pd.concat([df_stat, stat], axis=0)

df_stat.index = [str(y)+'_'+str(int(x/8)) for x,y in zip(range(len(df_stat)), df_stat.index)]
df_stat = df_stat.sort_index(axis=0)

dict_stat = {j:df_stat.iloc[i:i+n,:] for i,j in zip(range(0, n*8, n), ['25','50','75','count','max',
                                                                       'mean','min','std'])}


#%% Cluster descriptive statistics
def print_stat(prop, j):
    fig = plt.figure(figsize=(14,14))
    plt.title(prop, size=15)
    ax0 = fig.add_subplot(j,2,1)
    ax1 = fig.add_subplot(j,2,2)
    ax2 = fig.add_subplot(j,2,3)
    ax3 = fig.add_subplot(j,2,4)
    #ax4 = fig.add_subplot(j,2,5)
    ax0.bar(range(5),dict_stat[prop]['vszam'])
    ax0.set_title('vszam')
    ax1.bar(range(5),dict_stat[prop]['bevetel'])
    ax1.set_title('bevetel')
    ax2.bar(range(5),dict_stat[prop]['haszonkulcs'])
    ax2.set_title('haszonkulcs')
    ax3.bar(range(5),dict_stat[prop]['vnap'])
    ax3.set_title('vnap')
    #ax4.bar(range(5),dict_stat[prop]['ar'])
    #ax4.set_title('ar')


print_stat('mean', 2)
print_stat('count', 2)
print_stat('min', 2)
print_stat('max', 2)
