# 🏥 Application Streamlit - Prédiction d'Assurance Véhicule

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-green.svg)

## 📋 Description

Cette application Streamlit est dédiée à la **segmentation de marché dans le domaine de l'assurance véhicule**. Elle utilise des modèles de Machine Learning d'ensemble pour prédire si un client sera intéressé par une assurance véhicule.

Cette application a été développée dans le cadre du **TP2 IIA S6 2025-2026** par **HLDX - Henri Ledoux SAME**.

## 🚀 Fonctionnalités

### 1. Page d'Accueil
- Statistiques rapides sur le jeu de données
- Aperçu des données
- Instructions d'utilisation

### 2. Prédiction Interactive
- Formulaire de saisie des informations client
- Prédiction en temps réel
- Affichage de la probabilité d'intérêt
- Importance des features

### 3. Analyse Exploratoire
- Statistiques descriptives
- Distribution de la variable cible
- Analyse démographique (âge, genre, véhicule)
- Matrice de corrélation

### 4. Performance du Modèle
- Entraînement de modèles ML
- Métriques (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Visualisations (Matrice de confusion, Courbes ROC, Courbes Precision-Recall)
- Importance des features

### 5. Guide
- Explication des métriques
- Description des variables
- Recommandations

## 🤖 Modèles ML Disponibles

- **Random Forest** - Forêt aléatoire
- **Gradient Boosting** - Boosting de gradient
- **Voting Classifier** - Combinaison de plusieurs modèles
- **Stacking Classifier** - Empilement de modèles

## 📦 Installation

1. Clonez le projet ou téléchargez les fichiers

2. Installez les dépendances :
```
bash
pip install -r requirements.txt
```

3. Lancez l'application :
```
bash
streamlit run app.py
```

## 📊 Données

Le jeu de données utilisé contient les colonnes suivantes :

| Variable | Description |
|----------|-------------|
| `id` | Identifiant unique du client |
| `Gender` | Genre du client (Male/Female) |
| `Age` | Âge du client |
| `Driving_License` | Permis de conduire (0/1) |
| `Region_Code` | Code de la région |
| `Previously_Insured` | Déjà assuré (0/1) |
| `Vehicle_Age` | Âge du véhicule (< 1 Year, 1-2 Year, > 2 Years) |
| `Vehicle_Damage` | Dommages au véhicule (Yes/No) |
| `Annual_Premium` | Prime annuelle |
| `Policy_Sales_Channel` | Canal de vente |
| `Vintage` | Ancienneté du client (jours) |
| `Response` | Réponse cible (0/1) - Intérêt pour l'assurance |

## 🎯 Variables Cibles

- **Response** : 1 = Intéressé, 0 = Non interesado

## 📈 Métriques d'Évaluation

- **Accuracy** - Précision globale
- **Precision** - Précision des prédictions positives
- **Recall** - Rappel des positifs réels
- **F1-Score** - Moyenne harmonique de Precision et Recall
- **ROC-AUC** - Aire sous la courbe ROC

## 🛠️ Technologies

- **Streamlit** - Framework web Python
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Plotly** - Visualisations interactives
- **Scikit-learn** - Modèles de Machine Learning

## 📝 Licence

Ce projet a été développé à des fins éducatives dans le cadre du TP2 IIA S6 2025-2026.

---

Développé par **HLDX - Henri Ledoux SAME** avec ❤️ et Streamlit
