# Import des librairies
import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Inclusion Financière en Afrique",
    page_icon="☏"
)

# Fonction pour afficher une section avec titre et contenu
def section(titre, contenu):
    st.header(titre)
    st.write(contenu)

# Fonction pour afficher une image avec un titre en dessous
def image_with_caption(image_path, caption):
    img = Image.open(image_path)
    st.image(img, caption=caption, use_column_width=True)

# Fonction pour afficher un paragraphe justifié
def paragraphe(texte):
    st.write(f"<div style='text-align: justify'>{texte}</div>", unsafe_allow_html=True)

# Titre de page
st.title("Inclusion Financière en Afrique")

# Image illustrative de l'application
image_with_caption("Inclusion_finance/Inclusion-FINANCIERE.jpeg", " ")

# Description de l'application
paragraphe("""

Ce point de contrôle porte sur l'ensemble de données "Inclusion financière en Afrique", fourni dans le cadre de l'initiative d'inclusion financière en Afrique et hébergé par la plateforme «indi». L'ensemble de données comprend des informations démographiques sur environ 33 600 personnes en Afrique de l'Est, ainsi que les services financiers qu'elles utilisent.

L'objectif du modèle de machine learning est de prédire quels individus sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire. L'inclusion financière vise à garantir que les individus et les entreprises ont accès à des produits et services financiers utiles et abordables, tels que les transactions, les paiements, les économies, le crédit et l'assurance, livrés de manière responsable et durable.

""")


# Définition de la section "Fonctionnalités de l'application"
def fonctionnalites_application():
    st.header("Fonctionnalités de l'application")

    # Texte justifié en HTML
    justification_texte = """
    <div style="text-align:justify">
    La fonctionnalité de cette application est de fournir une solution basée sur le machine learning pour prédire 
    quelles personnes sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire en Afrique de l'Est. 
    L'ensemble de données utilisé contient des informations démographiques sur environ 33 600 individus de la région 
    et indique quels services financiers ils utilisent.L'objectif global est d'aider à promouvoir l'inclusion 
    financière en Afrique de l'Est en identifiant les populations qui pourraient bénéficier le plus de l'accès à 
    des services bancaires. L'inclusion financière vise à garantir que les individus et les entreprises ont accès 
    à des produits et services financiers qui répondent à leurs besoins de manière abordable, responsable et durable.
    </div>
    """
    st.markdown(justification_texte, unsafe_allow_html=True)

# Affichage de la section
fonctionnalites_application()

# Définition de la section avec justification CSS
def section(titre, texte):
    st.header(titre)
    st.markdown(f'<div style="text-align: justify;">{texte}</div>', unsafe_allow_html=True)

# Affichage de chaque section avec justification
section("Informations sur les données", "Cette étape consiste à communiquer les résultats de l'analyse des données de manière claire et compréhensible. Cela peut inclure la création de rapports, de visualisations ou de tableaux de bord pour présenter les conclusions de l'analyse. L'objectif est de fournir des informations exploitables aux parties prenantes et de soutenir la prise de décision basée sur les données.")

section("Exploration des données", "Cette étape consiste à explorer et à comprendre les données brutes. Elle inclut l'examen initial des données pour identifier les tendances, les modèles, les valeurs aberrantes et les relations entre les différentes variables. L'objectif principal de cette étape est de générer des hypothèses et des questions pour guider les analyses ultérieures.")

section("Manipulation des données", "Une fois que les données ont été explorées, la manipulation des données intervient pour nettoyer, transformer et préparer les données en vue de l'analyse. Cela peut inclure le traitement des valeurs manquantes, la normalisation des données, la création de nouvelles variables dérivées, etc. L'objectif est de rendre les données prêtes à être utilisées dans les modèles d'analyse ou les visualisations.")

section("Modélisation", "Cette étape implique la construction de modèles statistiques ou d'algorithmes d'apprentissage automatique pour répondre à des questions spécifiques ou résoudre des problèmes. Cela peut inclure l'utilisation de techniques telles que la régression, la classification, le clustering, etc. L'objectif est de créer des modèles prédictifs ou des représentations des données qui peuvent être utilisés pour prendre des décisions ou générer des insights.")

section("Contactez-Nous", "Prendre contact pour plus d'information")

section("À Propos De Nous", "Qui sommes nous et comment nous rejoindre")

# Lien vers les autres pages ou sections
st.subheader("Des liens hypertextes menant à d'autres pages.")

st.write("""
- [Informations](http://localhost:8502/Informations)
- [Exploration des données](http://localhost:8502/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8502/Manipulation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8502/Modele_prediction)
- [Data modélisation](http://localhost:8502/Data_modelisation)
""")
