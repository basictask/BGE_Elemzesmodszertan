#%% Esettanulmany
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#%% Adatok beolvasása
df = pd.read_csv('student_records.csv')


#%% Adatok előkészítése
training_features = df[['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']]

outcome_labels = df[['Recommend']]

# Numerikus és kategorikus változók szétválogatása
numeric_feature_names = ['ResearchScore', 'ProjectScore']
categoricial_feature_names = ['OverallGrade', 'Obedient']


#%% Numerikus jellemzők szerkesztése
ss = StandardScaler() # StandardScaler objektum

# Illesszük az adatokra 
ss.fit(training_features[numeric_feature_names])

# Transzformáljuk az adatokat a scaler szerint
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])


#%% Kategorikus változók szerkesztése
training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)

categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))


#%% Modellezés
# Regresszor létrehozása
lr = LogisticRegression()

# Függvény illesztése 
model = lr.fit(training_features, np.array(outcome_labels['Recommend']))


#%% Modell értékelés
# egyszerű értékelés a training adatokon
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels['Recommend'])

print('Accuracy:', float(accuracy_score(actual_labels, pred_labels))*100, '%')
print('Classification Stats:')
print(classification_report(actual_labels, pred_labels))


#%% Predikció élesben
# Új adatok beolvasása
new_data = pd.read_csv('new_data.csv', sep=';')
prediction_features = new_data[new_data.columns]


#%% Új adatok átalakítása
# méretezés
prediction_features[numeric_feature_names] = ss.transform(prediction_features[numeric_feature_names])

# kategória változók
prediction_features = pd.get_dummies(prediction_features, columns=categoricial_feature_names)

# hiányzó kategória oszlopok hozzáadása
current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
missing_features = set(categorical_engineered_features) - current_categorical_engineered_features

# nullák hozzáadása, mert az adathalmazban nem fordult elő minden jegyből
for feature in missing_features:    
    prediction_features[feature] = [0] * len(prediction_features)  

prediction_features.drop('Name', axis=1, inplace=True)


#%% Predikció új adatokon a modellel
predictions = model.predict(prediction_features)

new_data['Recommend'] = predictions
