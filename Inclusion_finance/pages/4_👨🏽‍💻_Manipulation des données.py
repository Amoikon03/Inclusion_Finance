import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    st.title("Manipulation Des Données")

    # Charger le fichier CSV
    df = pd.read_csv("Inclusion.csv")

    afficher_informations_generales(df)
    supprimer_doublons(df)
    traiter_valeurs_manquantes(df)
    enregistrer_donnees_pretraitees(df)


def afficher_informations_generales(df):
    st.markdown("<span style='color:black'>**-- Afficher des informations générales sur l'ensemble de données**</span>",
                unsafe_allow_html=True)
    st.write(df.head())
    st.write(df.info())


def supprimer_doublons(df):
    # Trouver les lignes dupliquées
    duplique = df.duplicated()
    duplicates = df[duplique]

    # Afficher les lignes dupliquées
    st.markdown("<span style='color:black'>**-- Lignes dupliquées**</span>", unsafe_allow_html=True)
    st.write(duplicates)

    st.markdown("<span style='color:black'>**-- Supprimer les valeurs dupliquées, s'ils existent**</span>", unsafe_allow_html=True)
    df.drop_duplicates(inplace=True)
    st.write("Les lignes dupliquées ont été supprimées avec succès.")


def traiter_valeurs_manquantes(df):
    st.markdown("<span style='color:black'>**-- Traiter les valeurs manquantes, si elles existent**</span>",
                unsafe_allow_html=True)
    st.write("Données manquantes dans le DataFrame :")
    st.write(df.isna().sum().T)

    st.markdown("<span style='color:black'>**--  Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles et par zéro pour les colonnes numériques**</span>", unsafe_allow_html=True)

    # Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles et par zéro pour les colonnes numériques
    fill_values = {col: df[col].mode()[0] if df[col].dtype == 'object' else 0 for col in df.columns}
    df.fillna(value=fill_values, inplace=True)

    st.write("**Afficher le DataFrame après le traitement des valeurs manquantes**")
    st.write(df.head())

    st.markdown(
        "<span style='color:black'>**-- Encoder les variables catégorielles**</span>",
        unsafe_allow_html=True)

    # Fonction pour encoder les variables catégorielles
    def preprocess_data(df):
        colonnes_categorielles = df.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for colonne in colonnes_categorielles:
            df[colonne] = label_encoder.fit_transform(df[colonne])
        return df

     # Encodage des variables catégorielles
    df_encoded = preprocess_data(df)

    # Affichage du DataFrame encodé
    #st.title("DataFrame après encodage")
    st.dataframe(df_encoded)

    # Affichage des types de données
    st.write("**Types de données dans le DataFrame**")
    st.write(df.dtypes.transpose())

    st.write("**Information**")

def enregistrer_donnees_pretraitees(df):
    df.to_pickle("Data_Frame.pkl")
    # Charger une image depuis votre système de fichiers
    image_path = "Inclusion_finance/megaphone.jpg"

    # Afficher l'image en arrière-plan
    # Redimensionner et afficher l'image en arrière-plan
    st.image(image_path, width=150, caption=' ', use_column_width=False)

    # Ajouter du texte au-dessus de l'image
    st.write(
    "<div style='color: #FFFFFF; background-color: #26474E; padding: 10px;'>"
    "<b>Les données prétraitées sont sauvegardées dans <span style='color: red;'>data_frame</span> au format pickle pour une utilisation ultérieure.</b>"
    "</div>",
    unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

st.write(' ')
st.write(' ')

# Lien vers les autres pages ou sections
st.subheader("Des liens hypertextes menant à d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8502/)
- [Informations](http://localhost:8502/Informations)
- [Exploration des données](http://localhost:8502/Exploration_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8502/Modele_prediction)
- [Data modélisation](http://localhost:8502/Data_modelisation)
""")
