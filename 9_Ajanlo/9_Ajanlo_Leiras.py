# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from ast import literal_eval


#%% Ajánló rendszer készítése a filmek leírása alapján
df = pd.read_csv('movies_metadata.csv', parse_dates=True)

df = df[['title','genres','runtime','vote_average','vote_count','release_date','overview','id']]

df.rename(columns={'release_date':'year'}, inplace=True)


#%% Gender és Dátum Oszlopok átalakítása
def convert_int(x):
    try:
        return datetime.datetime.strptime(str(x), '%Y-%m-%d').year
    except: 
        return 0

df['year'] = df['year'].apply(convert_int)

df['genres'] = df['genres'].apply(literal_eval)

df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


#%%
# TF-IDF vektorizáló beimportálása
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF objektum létrehozása és stopszavak kiküszöbölése
tfidf = TfidfVectorizer(stop_words='english')

#NaN kicserélése üres szavakra
df['overview'] = df['overview'].fillna('')

# Végső TF-IDF műtrix létrehozása
tfidf_matrix = tfidf.fit_transform(df['overview'])

print(tfidf_matrix.shape)


#%% Belső szorzat kiszámítása lineáris kernellel
from sklearn.metrics.pairwise import linear_kernel

# Koszinusz hasonlósági mátrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#%% Egy fordított leképezés létrehozása
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


#%% Függvény, ami egy filmet fogad paraméterként, és kiadja a javaslatokat 
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # A címnek megfelelő index lekérése
    idx = indices[title]

    # Páros hasonlósági pontszámok lekérése
    # és tuple-listává alakítása
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Filmek rendezése a koszinusz hasonlósági pontok alapján
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # A 10 leghasonlóbb film lekérése
    sim_scores = sim_scores[1:11]

    # Indexek kitranszformálása
    movie_indices = [i[0] for i in sim_scores]

    # Top 10 film lekérése 
    return df['title'].iloc[movie_indices]


#%% Az oroszlánkirályhoz javaslatokat lekérni
content_recommender('Star Wars')

content_recommender('The Lion King')

content_recommender('The Godfather')
