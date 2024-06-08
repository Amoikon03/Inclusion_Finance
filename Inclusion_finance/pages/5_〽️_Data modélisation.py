import streamlit as st
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def afficher_data_modelisation():
    # Warnings
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Titre de la page
    st.title("Data Modélisation")

    # Chargement des données
    df = pk.load(open("Data_Frame.pkl", "rb"))

    # Créer un bouton pour afficher un message d'information
    if st.checkbox("**Cliquez ici pour masquer l'information**", value=True):

        # Ajout du CSS personnalisé
        st.markdown(
            """
            <style>
            .custom-info {
                color: #FB7A25; /* Couleur du texte */
                background-color: #FEFEFF; /* Couleur du fond */
                font-size: 20px; /* Taille de la police */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Affichage du message avec la classe CSS personnalisée
        st.markdown(
            '<div class="custom-info">2 algorithmes de machine learning sont disponibles sur cette page</div>',
            unsafe_allow_html=True)

        # Ajout du CSS personnalisé
        st.markdown(
            """
            <style>
            .custom-success {
                color: #FB7A25; /* Couleur du texte */
                background-color: #100B00; /* Couleur du fond */
                font-size: 18px; /* Taille de la police */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Affichage du message avec la classe CSS personnalisée
        st.markdown(
            '''
            <div class="custom-success">
                <strong>Le résultat du modèle choisi sera affiché en bas de cette information.</strong>
                <br><strong>Sur cette page vous pouvez faire de la prédiction ou de la classification.</strong>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.write(' ')
        st.write(' ')

        st.write("""
         **La Regression Logistique**  Offre une interprétation claire des facteurs influençant la propension à posséder un compte bancaire, avec une mise en œuvre rapide et une facilité d'interprétation, soutenant ainsi une approche transparente et efficace de l'inclusion financière..
        \n **Le Random Forest** Idéal pour capturer les relations complexes entre les caractéristiques démographiques et financières, et robuste face aux valeurs manquantes, offrant ainsi une prédiction précise de l'utilisation des comptes bancaires..
        """)

    # Les modèles disponibles
    model = st.sidebar.selectbox("Choisissez un model",
                                 ["Regression Logistique", "Random Forest"])

    # ✂️ Selection et découpage des données
    seed = 123
    def select_split(dataframe):
        x = dataframe.drop('bank_account', axis=1)
        y = dataframe['bank_account']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    # Création des variables d'entrainement et test
    x_train, x_test, y_train, y_test = select_split(dataframe=df)
    x_train, x_test, y_train, y_test = select_split(dataframe=df)

    # ✏️ Afficher les graphiques de performance sans try et après avec try except
    def plot_perf(graphes, model, x_test, y_test, cmap='Blues', fontsize=10):
        if "Confusion matrix" in graphes:
            st.subheader("Matrice de confusion")
            try:
                y_pred = model.predict(x_test)  # Si le modèle n'est pas un estimateur sklearn
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 7))
                sns.set(font_scale=1.1)  # Ajuster l'échelle de la police
                sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap, cbar=True, annot_kws={"size": fontsize})
                plt.xlabel('Valeurs prédites', fontsize=12)
                plt.ylabel('Valeurs réelles', fontsize=12)
                plt.title('Matrice de confusion', fontsize=14)
                classes = ['Classe 0', 'Classe 1']  # Mettre à jour les classes si nécessaire
                plt.xticks(ticks=[0.5, 1.5], labels=classes, fontsize=12)
                plt.yticks(ticks=[0.5, 1.5], labels=classes, fontsize=12, rotation=0)
                st.pyplot(plt)
            except ValueError as ve:
                st.warning(f"Erreur lors de la création de la matrice de confusion : {str(ve)}")
            except Exception as e:
                st.warning(f"Une erreur inattendue s'est produite : {str(e)}")

        if "ROC Curve" in graphes:
            st.subheader("La courbe ROC (Receiver Operating Characteristic)")
            try:
                disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
                disp.plot()
                st.pyplot()
                st.info("Une courbe ROC idéale se rapproche du coin supérieur gauche du graphique, "
                        "ce qui indique un modèle avec une sensibilité élevée et un faible taux de faux positifs.")
            except Exception as e:
                st.warning(f"La courbe ROC ne peut pas être représenté avec les données du modèle: {str(e)}")

        if "Precision_Recall Curve" in graphes:
            st.subheader("La courbe de Précision Recall)")
            try:
                disp = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
                disp.plot()
                st.pyplot()

                st.info("La courbe PR met en lumière la capacité du modèle à classer correctement les échantillons positifs,"
                   "ce qui est crucial dans les situations où les classes sont fortement déséquilibrées.")

                st.markdown("""
                <div style="text-align: justify;">
                    Une AUC-PR proche de 1 indique un modèle parfait, où chaque prédiction positive est correcte et chaque prédiction négative est incorrecte.
                    Comme pour la courbe ROC, une courbe PR idéale se rapproche du coin supérieur droit du graphique, indiquant que le modèle atteint à la fois une haute précision et un rappel élevé pour un seuil de classification donné.
                    Une courbe PR idéale monte rapidement depuis l'origine, atteignant une haute précision pour un rappel relativement faible, ce qui signifie que le modèle peut bien classer les échantillons positifs dès le début de la prédiction.
                    Une courbe PR idéale est lisse, sans oscillations ni variations abruptes, ce qui montre que le modèle maintient une précision élevée même en rappelant un grand nombre d'échantillons positifs.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"La courbe de Précision Recall ne peut pas être représenté avec les données du modèle: {str(e)}")

    # 3️⃣ Regression logistique
    if model == "Regression logistique":
        st.sidebar.subheader("Les hyperparamètres du modéle")

        hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

        n_max_iter = st.sidebar.number_input("Choisir le nombre maximal d'itération", 100, 1000, step=10)

        graphes_perf = st.sidebar.multiselect("Choisir un ou des graphiques de performance du model à afficher",
                                              ["Confusion matrix", "ROC Curve", "Precision_Recall Curve"])

        if st.sidebar.button("Prédire", key="logistic_regression"):
            st.subheader("Résultat de la Regression logistique")

            # Initialiser le modèle
            model = LogisticRegression(C=hyp_c, max_iter=n_max_iter, random_state=seed)

            # Entrainer le modèle
            model.fit(x_train, y_train)

            # Prédiction du modèle
            y_pred = model.predict(x_test)

            # Calcul des metrics de performances
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Afficher les métrics
            st.write("Exactitude du modèle :", accuracy)
            st.write("Précision du modèle :", precision)
            st.write("Recall du modèle :", recall)

            # Afficher les graphiques de performances
            plot_perf(graphes_perf, model, x_test, y_test)

    # 4️⃣ Random Forest
    elif model == "Random Forest":
        st.sidebar.subheader("Hyperparamètres pour le modèle Random Forest")

        n_estimators = st.sidebar.number_input("Nombre d'estimateurs", 1, 1000, step=1)
        max_depth = st.sidebar.slider("Profondeur maximale de chaque arbre", 1, 20, 10)

        graphes_perf = st.sidebar.multiselect("Sélectionnez un ou plusieurs graphiques de performance à afficher",
                                              ["Confusion matrix", "ROC Curve", "Precision_Recall Curve"])

        if st.sidebar.button("Predict", key="random_forest"):
            st.subheader("Résultats du modèle Random Forest")

            # Initialize the Random Forest model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)

            # Train the model
            model.fit(x_train, y_train)

            # Make predictions
            y_pred = model.predict(x_test)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')

            # Display metrics
            st.write("Model Accuracy:", accuracy)
            st.write("Model Precision:", precision)
            st.write("Model Recall:", recall)

            # Display performance graphs
            plot_perf(graphes_perf, model, x_test, y_test)

    # 5️⃣ Support Vector Machin
    #elif model == "Support Vecteur Machine":
        # st.sidebar.subheader("Les hyperparamètres du modéle")

        # hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)
        # kernel = st.sidebar.selectbox("Choisir le kernel", ["linear", "poly", "rbf", "sigmoid"])

        # graphes_perf = st.sidebar.multiselect("Sélectionnez un ou plusieurs graphiques de performance à afficher",
                                              # ["Confusion matrix", "ROC Curve", "Precision_Recall Curve"])

        # if st.sidebar.button("Predict", key="svm"):
            # st.subheader("Résultats du modèle SVM")

            # Initialize the SVM model
            # model = SVC(C=hyp_c, kernel=kernel, probability=True)

            # Train the model
            # model.fit(x_train, y_train)

            # Make predictions
            # y_pred = model.predict(x_test)

            # Calculate performance metrics
            # accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred, average='micro')
            # recall = recall_score(y_test, y_pred, average='micro')

            # Display metrics
            # st.write("Model Accuracy:", accuracy)
            # st.write("Model Precision:", precision)
            # st.write("Model Recall:", recall)

            # Display performance graphs
            # plot_perf(graphes_perf, model, x_test, y_test)


if __name__ == "__main__":
    afficher_data_modelisation()

# Lien vers les autres pages ou sections
st.subheader("Des liens hypertextes menant à d'autres pages.")

st.write("""
- [Acceuil](http://localhost:8502/)
- [Informations](http://localhost:8502/Informations)
- [Exploration des données](http://localhost:8502/Exploration_des_donn%C3%A9es)
- [Manipulation des données](http://localhost:8502/Manipulation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8502/Modele_prediction)
""")