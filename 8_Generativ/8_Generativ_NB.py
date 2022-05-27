# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re

import warnings
warnings.filterwarnings('ignore')


#%% Adatok olvasása
df = pd.read_csv('spam.csv', encoding='latin')
print(df.shape)
df.head()


#%% Átalakítás-vizualizáció
df.drop([x for x in df.columns if 'Unnamed' in x], axis=1, inplace=True)
df.columns = ['class', 'text']

plt.figure(figsize=(6,6))
df.groupby('class').count().plot.bar(ylim=0)
plt.show()


#%% Töltsük le az angol nyelvben taláható felesleges szavakat, mint az "is", "at"
nltk.download('stopwords')


#%% Előfeldolgozás
stemmer = PorterStemmer() # Ugyanazon szavak más alakú előfordulását lecsökkenti, pl. "argue", "arguing", "argued" 
words = stopwords.words("english")

df['processedtext'] = df['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

print(df.shape)
print(df.head(10))


#%% Train-test szétválasztás
target = df['class']

X_train, X_test, y_train, y_test = train_test_split(df['processedtext'], target, test_size=0.30, random_state=100)

print(df.shape); print(X_train.shape); print(X_test.shape)


#%% Szavak vektorizálása: a szavak előfordulását gyakorisági vektorokká alakítja
# TF: Term-frequency: normalizált gyakoriság az egész dokumentumban
# IDF: Inverse Document Frequency: Csökkenti azoknak a szavaknak a súlyát, amik dokumentum-szerte mindenhol előfordulnak
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))

test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))

print(vectorizer_tfidf.get_feature_names()[:10])

print(train_tfIdf.shape); print(test_tfIdf.shape)


#%% Osztályozó létrehozása és tanítása
nb_classifier = MultinomialNB()

nb_classifier.fit(train_tfIdf, y_train)

pred2 = nb_classifier.predict(test_tfIdf) 
print(pred2[:10])


#%% Metrika 
accuracy_tfidf = metrics.accuracy_score(y_test, pred2)
print(accuracy_tfidf)

Conf_metrics_tfidf = metrics.confusion_matrix(y_test, pred2, labels=['ham', 'spam'])
print(Conf_metrics_tfidf)


#%% Próbáljuk ki ugyanezt az eljárást egy véletlen-erdővel
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 100)

classifier.fit(train_tfIdf, y_train)


#%% RF osztályozó kiértékelése
predRF = classifier.predict(test_tfIdf) 
print(predRF[:10])

# Calculate the accuracy score
accuracy_RF = metrics.accuracy_score(y_test, predRF)
print(accuracy_RF)

Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=['ham', 'spam'])
print(Conf_metrics_RF)
