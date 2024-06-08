# Importater les librairies
import streamlit as st
import pandas as pd

# Titre de la page
st.title("Les informations relatives à ces données")


# Chargement de données du fichier csv
df = pd.read_csv("Inclusion_finance/Inclusion.csv")


# Afficher les noms des données
st.markdown('<div style="background-color: yellow; padding: 10px;">Voici les données requises pour cette application</div>', unsafe_allow_html=True)
st.write(df.head())

# Afficher les noms des colonnes
st.subheader("**Informations concernant les noms des colonnes.**")

st.write("1 - **country** : Pays dans lequel se trouve la personne interrogée.")
st.write("2 - **year** : Année de l'enquête.")
st.write("3 - **uniqueid** : Identifiant unique pour chaque personne interrogée")
st.write("4 - **bank_account** : Indique si l'individu possède ou utilise un compte bancaireType de lieu : Rural, Urbain")
st.write("5 - **location_type** : Type de localisation où réside l'individu (rural ou urbain)")
st.write("6 - **cellphone_access** : Indique si l'individu a accès à un téléphone portable")
st.write("7 - **household_size** : Nombre de personnes vivant dans une maison")
st.write("8 - **age_of_respondent** : L'âge de la personne interrogée")
st.write("9 - **gender_of_respondent** : Sexe de la personne interrogée : homme, femme")
st.write("10 - **relationship_with_head** : Relation de la personne interrogée avec le chef de famille : chef de famille, conjoint, enfant, parent, autre parent, autres personnes non apparentées, ne sait pas")
st.write("11 - **marital_status** : Statut matrimonial de la personne interrogée : Marié/vivant ensemble, Divorcé/séparé, Veuf, Célibataire/jamais marié, Ne sait pas")
st.write("12 - **education_level** : Niveau d'éducation le plus élevé : Pas d'éducation formelle, Enseignement primaire, Enseignement secondaire, Formation professionnelle/spécialisée, Enseignement supérieur, Autre/ne sait pas/RTA")
st.write("13 - **job_type** : Type d'emploi de la personne interrogée : Agriculture et pêche, travailleur indépendant, employé officiel du gouvernement, employé officiel du secteur privé, employé informel, dépendant des envois de fonds, dépendant du gouvernement, autre revenu, pas de revenu, ne sait pas/refuse de répondre.")


# Instruction
# Utilisation de HTML pour styliser le texte avec un fond jaune
st.write('NB : **Ces informations donnent un aperçu des données contenues dans chaque colonne de lensemble de données sur linclusion financière en Afrique**')

# Lien vers les autres pages ou sections
st.subheader("Des liens hypertextes menant à d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8502/)
- [Exploration des donées](http://localhost:8502/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8502/Manipulation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8502/Modele_prediction)
- [Data modélisation](http://localhost:8502/Data_modelisation)
""")
