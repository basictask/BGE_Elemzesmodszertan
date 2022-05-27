# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


#%% u.user fájl betöltése
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')


#%% u.item fájl betöltése
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

# Minden információ törlése kivéve movie_id és title
movies = movies[['movie_id', 'title']]


#%% u.data fájl betöltése
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings = ratings.drop('timestamp', axis=1)


#%% Train-teszt szétválasztás
from sklearn.model_selection import train_test_split

X = ratings.copy()
y = ratings['user_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)


#%% Függvény ami kiszámítja a gyök-eltérés négyzetösszeget
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#%% Az alap értékelést állítsuk 3-ra
def baseline(user_id, movie_id):
    return 3.0


#%% Függvény ami kiszámítja az RMSE-t adott modellel a teszt halmazon
def score(cf_model):
    iterable = zip(X_test['user_id'], X_test['movie_id'])
    
    # Minden Tuple-hez értékelés predikció hozzárendelése
    y_pred = np.array([cf_model(user, movie) for user, movie in iterable])
    
    # Kivonatolni a valós értékeléseket
    y_true = np.array(X_test['rating'])
    
    # Végső RMSE visszatérítése
    return rmse(y_true, y_pred)


# kipróbálás alap modellel
print("Alap modell:", score(baseline))


#%% Felhasználó alapú kollaboratív szűrés ####################################
# Értékelési mátrix létrehozása
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')


#%% Kollaboratív szűrés az átlagos értékelésekkel
def cf_user_mean(user_id, movie_id):
    
    # Megnézni, hogy a movie_id létezik-e a mátrixban
    if movie_id in r_matrix:
        # Filmre adott átlagos értékelések ellenőrzése
        mean_rating = r_matrix[movie_id].mean()
    
    else:
        # Alap értékelés 3-ra állítása
        mean_rating = 3.0
    
    return mean_rating


print('Felhasználói átlagok alapján: ', score(cf_user_mean))


#%% Kollaboratív szűrés súlyozott átlagokkal
# Dummy mátrix létrehozása nulla értékekkel a hiányzók helyén
r_matrix_dummy = r_matrix.copy().fillna(0)


#%% Koszinusz hasonlóság kiszámítása a dummy mátrixon
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)    

cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)

cosine_sim.head(10)


#%% Súlyozott felhasználói átlagok alapján kollaboratív szűrés
def cf_user_wmean(user_id, movie_id):
    # Létezik-e a film a dummy mátrixban
    wmean_rating = 3.0
    if movie_id in r_matrix:
        
        # A felhasználó és a többi felhasználó közötti koszinusz hasonlóság lekérése
        sim_scores = cosine_sim[user_id]
        
        # A film felhasználói értékelésének lekérése
        m_ratings = r_matrix[movie_id]
        
        # NaN indexek kiindexelése
        idx = m_ratings[m_ratings.isnull()].index
        
        # NaN értékek eldobása
        m_ratings = m_ratings.dropna()
        
        # A megfelelő koszinusz-hasonlósági pontok eldobása
        sim_scores = sim_scores.drop(idx)
        
        # Végső súlyozott átlag kiszámítása
        wmean_rating = np.dot(sim_scores, m_ratings) / sim_scores.sum()
    
    return wmean_rating

print(score(cf_user_wmean))


#%% Demográfiai megközelítés #################################################
# Az eredeti movies dataset összekapcsolása a felhasználókkal 
merged_df = pd.merge(X_train, users)


#%% Nem szerinti átlagos értékeléseket kiszámítani
gender_mean = merged_df[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()

# A users táblában az indexet a user_id-ra állítani
users = users.set_index('user_id')


#%% Nem alapú kollaboratív szűrés átlagos értékelések felhasználásával
def cf_gender(user_id, movie_id):
    
    # Létezik-e a film?
    if movie_id in r_matrix:
        # A felhasználó nemének beazonosítása
        gender = users.loc[user_id]['sex']
        
        # Az ő neme értékelte a filmet?
        if gender in gender_mean[movie_id]:
            
            # Az ő neme szerinti értékelések leszűrése
            gender_rating = gender_mean[movie_id][gender]
        
        else:
            gender_rating = 3.0
    
    else:
        # Az alap értékelés 3-ra állítása
        gender_rating = 3.0
    
    return gender_rating


print(score(cf_gender))


#%% Demográfiai megközelítés foglalkozásokkal
# Átlagos értékelés nem és foglalkozás szerint
gen_occ_mean = merged_df[['sex', 'rating', 'movie_id', 'occupation']].pivot_table(
    values='rating', index='movie_id', columns=['occupation', 'sex'], aggfunc='mean')


#%% Nem és foglalkozás alapú kollaboratív szűrés
def cf_gen_occ(user_id, movie_id):
    if movie_id in gen_occ_mean.index:
        user = users.loc[user_id]
        gender = user['sex']
        occ = user['occupation']
        
        # A foglalkozás értékelte a filmet?
        if occ in gen_occ_mean.loc[movie_id]:
            
            # A nem értékelte a filmet?
            if gender in gen_occ_mean.loc[movie_id][occ]:
                
                # Szükséges értékelés lekérése
                rating = gen_occ_mean.loc[movie_id][occ][gender]
                
                # 3 alapértékre állítás
                if np.isnan(rating):
                    rating = 3.0
                return rating
    return 3.0

print(score(cf_gen_occ))


#%% Modellalapú megközelítés ################################################
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate

# A Reader objektum segít bejárni a string fájlokat, dataframeket
reader = Reader()

# A szűréshez szükséges dataset létrehozása
data = Dataset.load_from_df(ratings, reader)

# KNN objektum létrehozása
knn = KNNBasic()

# KNN kiértékelése
cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


#%%
#Import SVD
from surprise import SVD

#Define the SVD algorithm object
svd = SVD()

#Evaluate the performance in terms of RMSE
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
