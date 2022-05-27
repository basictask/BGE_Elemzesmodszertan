# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


#%% Egyszerű ajánló rendszer
df = pd.read_csv('movies_metadata.csv')
df.head()


#%%# Annak a kiszámítása, hogy a 80. percentilis film mennyi szavazatot kapott
m = df['vote_count'].quantile(0.80)

# Csak 45 percnél hosszabb és 300 percnél rövidebb filmeket vegyen figyelembe
q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]

# Csak m-nél magasabb szavazatú filmeket vegyen figyelembe
q_movies = q_movies[q_movies['vote_count'] >= m]

print('bekerült filmek száma', q_movies.shape)

# Átlagos szavazat kiszámítása
C = df['vote_average'].mean()


#%% IMDB súlyozott átlag kiszámítása minden filmre
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


#%% Pontszám kiszámítása az előbb definiált függvénnyel
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


#%% Csökkenő sorrendbe rendezés és kiíratás
q_movies = q_movies.sort_values('score', ascending=False)

# top 25 kiíratása
print(q_movies[['title', 'vote_count', 'vote_average', 'score', 'runtime']].head(25))
