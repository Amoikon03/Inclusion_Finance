import streamlit as st
import pickle
import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Charger le DataFrame à partir d'un fichier pickle
def load_dataframe(filepath):
    with open(filepath, 'rb') as file:
        dataframe = pickle.load(file)
    return dataframe


# Fonction pour entraîner les modèles
def train_models(data):
    # Sélectionner les caractéristiques et la cible
    X = data.drop('bank_account', axis=1)
    y = data['bank_account']

    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle Random Forest
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)

    # Entraîner le modèle de régression logistique
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)

    return model_rf, model_lr, le, X_train.columns


# Fonction pour préparer les caractéristiques
def prepare_features(features, train_columns, le):
    # Séparer les colonnes numériques et catégorielles
    numeric_cols = features.select_dtypes(include=['int', 'float']).columns
    categorical_cols = features.select_dtypes(include=['object']).columns

    # Imputer les données numériques avec la moyenne
    if not numeric_cols.empty:
        imputer_numeric = SimpleImputer(strategy='mean')
        filled_numeric = imputer_numeric.fit_transform(features[numeric_cols])
        features[numeric_cols] = filled_numeric

    # Imputer les données catégorielles avec la valeur la plus fréquente
    if not categorical_cols.empty:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        filled_categorical = imputer_categorical.fit_transform(features[categorical_cols])
        features[categorical_cols] = filled_categorical

        # Encoder les variables catégorielles
        for col in categorical_cols:
            features[col] = le.fit_transform(features[col].astype(str))

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in train_columns:
        if col not in features.columns:
            features[col] = 0  # Ou une autre valeur par défaut appropriée

    # Assurez-vous que les colonnes sont dans le même ordre que lors de l'entraînement
    features = features[train_columns]
    return features


# Fonction pour prédire avec Random Forest
def predict_random_forest(model, features):
    # Prédire avec le modèle
    prediction = model.predict_proba(features)[:, 1]
    return prediction


# Fonction pour prédire avec la régression logistique
def predict_logistic_regression(model, features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    prediction = model.predict_proba(scaled_features)[:, 1]
    return prediction

# Définir le titre de l'application avec des options de personnalisation
titre = "Prédiction d'Inclusion Financière en Afrique"

# Définir les options de personnalisation
couleur_texte = "#47EAD0"
couleur_fond = "#0C0F0A"  # Couleur en format hexadécimal

# Définir le style CSS pour le titre
style_titre = f"color: {couleur_texte}; background-color: {couleur_fond}; padding: 10px;"

# Afficher le titre personnalisé
st.markdown(f'<h1 style="{style_titre}">{titre}</h1>', unsafe_allow_html=True)

st.write(" ")
st.write(" ")

# Charger le DataFrame à partir du fichier pickle
data = load_dataframe("Data_Frame.pkl")

# Créer une colonne latérale pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    age = st.number_input("Âge", min_value=0, max_value=120, step=1)
    education = st.selectbox("Niveau d'éducation", ["Primaire", "Secondaire", "Université"])
    monthly_income = st.number_input("Revenu mensuel (fcfa)", min_value=0)
    internet_access = st.radio("Accès à Internet", ["Oui", "Non"])
    occupation = st.selectbox("Occupation", ["Employé", "Indépendant", "Étudiant", "Sans emploi", "Autre"])
    marital_status = st.selectbox("Situation matrimoniale", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"])
    dependents = st.number_input("Nombre de personnes à charge", min_value=0, step=1)
    # Utiliser une liste déroulante pour le pays
    country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzanie", "Ouganda"])

    # Créer une liste de toutes les villes
    all_cities = {
        "Kenya": ["Nairobi", "Mombasa", "Kisumu"],
        "Rwanda": ["Kigali", "Butare", "Gisenyi"],
        "Tanzanie": ["Dar es Salaam", "Dodoma", "Mwanza"],
        "Ouganda": ["Kampala", "Gulu", "Lira"]
    }

    # Utiliser une condition pour sélectionner la liste de villes en fonction du pays sélectionné
    cities = all_cities.get(country, ["Aucune ville disponible pour ce pays"])

    # Créer une liste de villes en fonction du pays sélectionné
    if country == "Kenya":
        cities = ["Nairobi", "Mombasa", "Kisumu"]
    elif country == "Rwanda":
        cities = ["Kigali", "Butare", "Gisenyi"]
    elif country == "Tanzanie":
        cities = ["Dar es Salaam", "Dodoma", "Mwanza"]
    elif country == "Ouganda":
        cities = ["Kampala", "Gulu", "Lira"]
    else:
        cities = ["Aucune ville disponible pour ce pays"]

    # Afficher la liste de villes dans une liste déroulante
    city = st.selectbox("Ville", cities)

    # Créer une liste des relations possibles avec le chef de famille
    relations_with_head = ["Époux/Épouse", "Fils/Fille", "Petit-fils/Petite-fille", "Autre membre de la famille",
                           "Non-membre de la famille"]

    # Utiliser une liste déroulante pour la relation avec le chef de famille
    relationship_with_head = st.selectbox("Relation avec le chef de famille", relations_with_head)

# Création d'un dictionnaire avec les paramètres
params = {
    'age': age,
    'education': education,
    'monthly_income': monthly_income,
    'internet_access': internet_access,
    'occupation': occupation,
    'marital_status': marital_status,
    'dependents': dependents,
    'country': country,
    'city': city,
    'relationship_with_head': relationship_with_head
}

# Convertir le dictionnaire en DataFrame
features = pd.DataFrame([params])

# Charger les modèles, l'encodeur et les colonnes d'entraînement
model_rf, model_lr, le, train_columns = train_models(data)

# Préparer les caractéristiques
features = prepare_features(features, train_columns, le)

# Prédire avec Random Forest
prediction_rf = predict_random_forest(model_rf, features)

# Prédire avec la régression logistique
prediction_lr = predict_logistic_regression(model_lr, features)

# Affichage des résultats des prédictions
probabilite_rf = prediction_rf[0]
probabilite_lr = prediction_lr[0]

# Formater les probabilités avec deux décimales
format_proba = lambda proba: f"{proba:.2f}"

# Affichage des probabilités pour chaque modèle
st.write(f"Probabilité d'avoir un compte bancaire (Random Forest) : {format_proba(probabilite_rf)}")
st.write(f"Probabilité d'avoir un compte bancaire (Régression Logistique) : {format_proba(probabilite_lr)}")


# Lien vers les autres pages ou sections
st.subheader("Des liens hypertextes menant à d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8502/)
- [Informations](http://localhost:8502/Informations)
- [Exploration des données](http://localhost:8502/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8502/Manipulation_des_donn%C3%A9es)
- [Data modélisation](http://localhost:8502/Data_modelisation)
""")