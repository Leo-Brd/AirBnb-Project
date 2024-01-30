# Projet AirBnb Leo, Antoine, Aziz

import pandas as pd
import numpy as np

CSV_PATH = 'res/paris_airbnb.csv'

# Lecture du dataset
print('--- Lecture du dataset')
df = pd.read_csv(CSV_PATH)
print('La première valeur est :', df.head(1), "\n")

# Distance euclidienne
print('\n--- Distance euclidienne')
first_distance = np.abs(3 - df.at[0, "accommodates"])
print('La première distance est :', first_distance, "\n")

# Calculer la distance pour toutes les observations
print('\n--- Calculer la distance pour toutes les observations')
df['distance'] = df['accommodates'].apply(lambda x: np.abs(x - 3))
print('Le nombre de distances calculées:', df['distance'].value_counts(), "\n")

# b) Randomisation et Tri
print('\n--- b) Randomisation et Tri')
np.random.seed(1)
random_indices = np.random.permutation(df.index)
df = df.loc[random_indices]
df = df.sort_values('distance')
print('10 premières valeurs triées:', df['price'].head(10), "\n")

# Prix moyen
print('\n--- Prix moyen')
df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].astype(float)
df['price'] = df['price'].apply(lambda x: x * 0.9)
mean_price = df['price'].head(5).mean()
print('Résultat:', mean_price)

# Fonction pour faire des prédictions
print('\n# Fonction pour faire des prédictions')

def predict_price(new_listing):
    temp_df = df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.iloc[:5]['price']
    predicted_price = nearest_neighbors.mean()
    return predicted_price

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)

print("Prix suggéré pour l'accueil d'1 personne:", acc_one, "\n")
print("Prix suggéré pour l'accueil de 2 personnes:", acc_two, "\n")
print("Prix suggéré pour l'accueil de 4 personnes:", acc_four, "\n")

# Fonction pour faire des prédictions
print('\n# Fonction pour faire des prédictions')


def predict_price2(new_listing, new_beds, new_bedrooms):
    temp_df = df.copy()
    temp_df['distance'] = np.sqrt((temp_df['accommodates'] - new_listing)**2 +
                                  (temp_df['beds'] - new_beds)**2 +
                                  (temp_df['bedrooms'] - new_bedrooms)**2)
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.iloc[:5]['price']
    predicted_price = nearest_neighbors.mean()
    return predicted_price


acc_one = predict_price2(1, 1, 1)
acc_two = predict_price2(2, 2, 1)
acc_four = predict_price2(4, 2, 2)

print("Prix suggéré pour un accueil d'1 personne, avec 1 lit et 1 chambre :", acc_one, "\n")
print("Prix suggéré pour un accueil de 2 personnes, avec 2 lits et 1 chambre :", acc_two, "\n")
print("Prix suggéré pour un accueil de 4 personnes, avec 2 lits et 2 chambres :", acc_four, "\n")

