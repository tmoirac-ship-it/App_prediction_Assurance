"""
🏥 Application Streamlit - Prédiction d'Assurance Véhicule
TP2 IIA S6 - Modèles ML d'Ensemble
=================================================================
Interface Streamlit pour le déploiement de modèles ML d'ensemble
Version avec design moderne et interactif

Utilisation:
    streamlit run app.py
=================================================================
"""

# ============================================================================
# IMPORTATION DES LIBRAIRIES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier, 
    AdaBoostClassifier, 
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix, 
    roc_curve, auc,
    precision_recall_curve, 
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="🚗 Prédiction Assurance Véhicule - TP2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## 🚗 Prédiction d'Assurance Véhicule
        
        Application de Machine Learning pour prédire si un client 
        est susceptible d'être intéressé par une assurance véhicule.
        
        Version: TP2 IIA S6 2025-2026
        """
    }
)

# ============================================================================
# STYLE CSS PERSONNALISÉ - NOUVEAU DESIGN
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Style général - THÈME CLAIR MODERNE */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Headers principaux */
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Couleurs personnalisées - NOUVELLE PALETTE TEAL/CYAN */
    .title-gradient {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    /* Nouvelle sidebar - fond clair élégant */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid #00b894;
    }
    
    /* Boutons personnalisés - Style moderne */
    .stButton>button {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.5);
    }
    
    /* Cartes modernes avec ombres */
    .modern-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #00b894;
    }
    
    /* Cartes彩色 avec gradient */
    .gradient-card {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        border-radius: 20px;
        padding: 30px;
        color: white;
        box-shadow: 0 10px 40px rgba(0, 184, 148, 0.3);
    }
    
    /* Animation slide-in */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Divider stylisé */
    hr {
        margin: 30px 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #00b894 0%, #00cec9 50%, #6c5ce7 100%);
        border-radius: 2px;
    }
    
    /* Metrics avec nouveau style */
    div.stMetric {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00b894 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #636e72 !important;
    }
    
    /* Headers colorés */
    h1, h2, h3, h4, h5, h6 {
        color: #2d3436 !important;
    }
    
    /* Enhanced input fields */
    .stSelectbox, .stNumberInput, .stTextInput {
        background: white;
        border-radius: 10px;
    }
    
    /* Tab styling moderne */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white !important;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
    }
    
    /* Spinner custom */
    .stSpinner > div {
        border: 4px solid #00b894;
        border-top: 4px solid #00cec9;
    }
    
    /* Alert styling */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        border-radius: 10px;
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style pour les expanders */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #00b894;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data
def load_data_from_file(uploaded_file):
    """Charger les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("❌ Format de fichier non supporté. Utilisez CSV ou Excel.")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

@st.cache_data
def load_default_data():
    """Charger les données par défaut"""
    return pd.read_csv('merged_dataset.csv')

def preprocess_data(df, target_col='Response'):
    """Prétraiter les données avec gestion des variables catégorielles"""
    data = df.copy()
    
    # Supprimer les colonnes non nécessaires
    if 'id' in data.columns:
        data = data.drop('id', axis=1)
    if 'dataset_type' in data.columns:
        data = data.drop('dataset_type', axis=1)
    
    # Mapper la cible si nécessaire
    if data[target_col].dtype == 'object':
        data[target_col] = data[target_col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
    
    # Identifier les colonnes
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    return data, categorical_cols, numerical_cols

def train_ensemble_model(X, y, model_type='voting'):
    """Entraîner différents types de modèles"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèles
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
    )
    
    lr = LogisticRegression(
        random_state=42, class_weight='balanced', max_iter=1000, C=0.5
    )
    
    if model_type == 'voting':
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft', n_jobs=-1
        )
    elif model_type == 'stacking':
        ensemble = StackingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            final_estimator=LogisticRegression(class_weight='balanced'), cv=5
        )
    else:
        ensemble = rf
    
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
    }
    
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='f1')
    
    return ensemble, scaler, X_test_scaled, y_test, y_pred, y_proba, metrics, cv_scores

def create_feature_importance_plot(model, feature_names):
    """Créer un graphique d'importance des features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_estimators_'):
        importances = model.named_estimators_['rf'].feature_importances_
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df, x='importance', y='feature',
        orientation='h', title="📊 Importance des Caractéristiques",
        color='importance', color_continuous_scale='Teal',
        text='importance'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template="plotly_white", height=500,
        xaxis_title="Importance", yaxis_title="Features",
        font=dict(family="Poppins")
    )
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    """Créer une matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=['Prédit: Non', 'Prédit: Oui'],
        y=['Réel: Non', 'Réel: Oui'],
        colorscale='Teal', text=cm, texttemplate='%{text}',
        textfont={"size": 25}, showscale=False
    ))
    
    fig.update_layout(title="🎯 Matrice de Confusion", template="plotly_white", height=400)
    return fig

def create_roc_curve_plot(y_true, y_proba):
    """Créer la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='#00b894', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Aléatoire', line=dict(color='#e17055', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="📈 Courbe ROC", xaxis_title="FPR",
        yaxis_title="TPR", template="plotly_white", height=400,
        legend=dict(x=0.7, y=0.1), font=dict(family="Poppins")
    )
    return fig

def create_precision_recall_plot(y_true, y_proba):
    """Créer la courbe Precision-Recall"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        name=f'PR (AP = {avg_precision:.3f})',
        line=dict(color='#00cec9', width=3)
    ))
    
    fig.update_layout(
        title="📉 Courbe Precision-Recall", xaxis_title="Recall",
        yaxis_title="Precision", template="plotly_white", height=400,
        font=dict(family="Poppins")
    )
    return fig

# ============================================================================
# SIDEBAR - CONFIGURATION MODERNISÉE
# ============================================================================

def create_sidebar():
    """Crée la barre latérale avec nouveau design"""
    
    # Logo/Titre dans sidebar
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 25px 10px;'>
        <div style='font-size: 50px;'>🚗</div>
        <h2 style='color: #00b894; margin: 10px 0; font-weight: 700;'>AUTO-ASSUR</h2>
        <p style='color: #636e72; font-size: 12px;'>ML Prédictif</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Source de données
    st.sidebar.markdown("### 📂 Source de Données")
    data_source = st.sidebar.radio(
        "Sélectionner:", ["Dataset défaut", "Importer fichier"]
    )
    
    df = None
    
    if data_source == "Importer fichier":
        uploaded_file = st.sidebar.file_uploader(
            "Charger CSV/Excel", type=['csv', 'xlsx']
        )
        if uploaded_file:
            df = load_data_from_file(uploaded_file)
            if df is not None:
                st.sidebar.success(f"✅ {uploaded_file.name}")
    else:
        df = load_default_data()
        st.sidebar.info("📊 Dataset chargé")
    
    st.sidebar.markdown("---")
    
    # Paramètres du modèle
    st.sidebar.markdown("### ⚙️ Modèle ML")
    
    model_type = st.sidebar.selectbox(
        "Algorithme",
        ['voting', 'stacking', 'random_forest'],
        format_func=lambda x: {
            'voting': '🗳️ Voting',
            'stacking': '🧱 Stacking',
            'random_forest': '🌲 Random Forest'
        }[x]
    )
    
    st.sidebar.markdown("---")
    
    # Navigation avec icons
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Pages:",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse", "📈 Performance", "❓ Guide"]
    )
    
    st.sidebar.markdown("---")
    
    # Badge
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                border-radius: 15px; padding: 20px; text-align: center;'>
        <p style='color: white; margin: 0; font-weight: 600;'>TP2 IIA S6</p>
        <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 5px 0;'>2025-2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    return df, model_type, page

# ============================================================================
# PAGE D'ACCUEIL - NOUVEAU DESIGN
# ============================================================================

def show_homepage():
    """Affiche la page d'accueil avec nouveau design"""
    
    # Hero section moderne
    st.markdown("""
    <div style='text-align: center; padding: 50px 20px;' class='slide-in'>
        <div style='font-size: 80px; margin-bottom: 20px;'>🚗✨</div>
        <h1 class='title-gradient'>Prédiction Assurance Véhicule</h1>
        <p style='font-size: 1.3rem; color: #636e72; max-width: 600px; margin: 20px auto;'>
            Intelligence Artificielle pour identifier les clients intéressés par une assurance véhicule
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cartes de fonctionnalités
    st.markdown("### ✨ Fonctionnalités")
    
    cols = st.columns(4)
    features = [
        ("📊", "Analyse Interactive", "Visualisez vos données"),
        ("🤖", "ML d'Ensemble", "Modèles avancés"),
        ("📈", "Métriques", "Évaluation complète"),
        ("🔮", "Prédiction", "Testez en temps réel")
    ]
    
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class='modern-card' style='text-align: center; padding: 20px;'>
                <div style='font-size: 40px;'>{icon}</div>
                <h4 style='color: #00b894; margin: 10px 0;'>{title}</h4>
                <p style='color: #636e72; font-size: 0.9rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistiques
    st.markdown("### 📊 Aperçu des Données")
    
    try:
        df = load_default_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Clients", f"{len(df):,}")
        with col2:
            if 'Response' in df.columns:
                interested = (df['Response'] == 1).sum()
                st.metric("✅ Intéressés", f"{interested:,}")
        with col3:
            if 'Response' in df.columns:
                rate = (df['Response'] == 1).mean() * 100
                st.metric("📈 Taux", f"{rate:.1f}%")
        with col4:
            st.metric("📋 Features", df.shape[1] - 2)
        
        # Preview
        st.markdown("### 👁️ Aperçu")
        st.dataframe(
            df.head(8), 
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
    
    # Guide
    st.markdown("---")
    st.markdown("""
    <div class='modern-card slide-in'>
        <h3 style='color: #00b894;'>🚀 Comment utiliser</h3>
        <ol style='line-height: 2; color: #2d3436;'>
            <li>Sélectionnez un modèle dans la barre latérale</li>
            <li>Utilisez <b>"Prédiction"</b> pour prédire l'intérêt d'un client</li>
            <li>Consultez <b>"Analyse"</b> pour explorer les données</li>
            <li>Visualisez les performances dans <b>"Performance"</b></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE PRÉDICTION - NOUVEAU DESIGN
# ============================================================================

def show_prediction_page(df, model_type):
    """Affiche la page de prédiction avec nouveau design"""
    
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;' class='slide-in'>
        <div style='font-size: 60px;'>🔮</div>
        <h1 class='title-gradient'>Prédiction en Temps Réel</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Entraîner si pas encore fait
        model_key = f"model_{model_type}"
        
        if model_key not in st.session_state:
            with st.spinner('🔧 Entraînement du modèle...'):
                data_processed, cat_cols, num_cols = preprocess_data(df)
                
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                    le_dict[col] = le
                
                X = data_processed.drop('Response', axis=1)
                y = data_processed['Response']
                
                model, scaler, X_test, y_test, y_pred, y_proba, metrics, cv_scores = train_ensemble_model(X, y, model_type=model_type)
                
                st.session_state[model_key] = {
                    'model': model, 'scaler': scaler,
                    'le_dict': le_dict, 'X': X
                }
            st.success("✅ Modèle prêt!")
        else:
            model_data = st.session_state[model_key]
            model = model_data['model']
            scaler = model_data['scaler']
            le_dict = model_data['le_dict']
            X = model_data['X']
        
        # Formulaire
        st.markdown("### 📝 Informations du Client")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("#### 👤 Profil")
                gender = st.selectbox("Genre", ['Male', 'Female'])
                age = st.number_input("Âge", 18, 100, 35)
                driving_license = st.selectbox("Permis", [1, 0], format_func=lambda x: "Oui" if x else "Non")
                region = st.number_input("Région", 0, 60, 28)
            
            st.markdown("#### 🚗 Véhicule")
            vehicle_age = st.selectbox("Âge Véhicule", ['< 1 Year', '1-2 Year', '> 2 Years'])
            vehicle_damage = st.selectbox("Dommages", ['Yes', 'No'])
        
        with col2:
            st.markdown("#### 🛡️ Assurance")
            previously_insured = st.selectbox("Déjà Assuré", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            
            st.markdown("#### 💵 Finance")
            premium = st.number_input("Prime (€)", 1000, 200000, 30000, 1000)
            
            st.markdown("#### 📞 Contact")
            channel = st.selectbox("Canal", [26, 152, 124, 160, 156])
            vintage = st.number_input("Vintage (jours)", 0, 300, 200)
        
        st.markdown("---")
        
        # Bouton de prédiction
        if st.button("🔮 Prédire l'Intérêt", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'Gender': [gender], 'Age': [age], 'Driving_License': [driving_license],
                'Region_Code': [region], 'Previously_Insured': [previously_insured],
                'Vehicle_Age': [vehicle_age], 'Vehicle_Damage': [vehicle_damage],
                'Annual_Premium': [premium], 'Policy_Sales_Channel': [channel], 'Vintage': [vintage]
            })
            
            for col in input_data.columns:
                if col in le_dict:
                    try:
                        input_data[col] = le_dict[col].transform([input_data[col].iloc[0]])[0]
                    except:
                        input_data[col] = 0
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            # Résultat
            st.markdown("### 🎯 Résultat")
            
            if prediction == 1:
                st.markdown(f"""
                <div class='gradient-card' style='text-align: center;'>
                    <div style='font-size: 60px;'>✅</div>
                    <h2 style='margin: 15px 0;'>Intéressé!</h2>
                    <p style='font-size: 1.2rem;'>Probabilité: {proba[1]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='gradient-card' style='background: linear-gradient(135deg, #e17055 0%, #d63031 100%); text-align: center;'>
                    <div style='font-size: 60px;'>❌</div>
                    <h2 style='margin: 15px 0;'>Non Intéressé</h2>
                    <p style='font-size: 1.2rem;'>Probabilité: {proba[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance
            st.markdown("### 📊 Impact")
            fig = create_feature_importance_plot(model, X.columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Veuillez charger des données")

# ============================================================================
# PAGE ANALYSE - NOUVEAU DESIGN
# ============================================================================

def show_analysis_page(df):
    """Affiche la page d'analyse avec nouveau design"""
    
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;' class='slide-in'>
        <div style='font-size: 60px;'>📊</div>
        <h1 class='title-gradient'>Analyse Exploratoire</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        tabs = st.tabs(["📈 Stats", "🎯 Target", "👥 Démo", "🔗 Corrélation"])
        
        with tabs[0]:
            st.markdown("### Statistiques Descriptives")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if 'id' in numeric_df.columns:
                numeric_df = numeric_df.drop('id', axis=1)
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        with tabs[1]:
            st.markdown("### Variable Cible")
            if 'Response' in df.columns:
                fig = px.pie(
                    df, names=df['Response'].map({0: 'Non Intéressé', 1: 'Intéressé'}),
                    title="Distribution", color_discrete_sequence=['#e17055', '#00b894']
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Total", len(df))
                col2.metric("Taux positif", f"{(df['Response']==1).mean()*100:.1f}%")
        
        with tabs[2]:
            st.markdown("### Analyse Démographique")
            
            if 'Age' in df.columns and 'Response' in df.columns:
                fig = px.box(
                    df, x='Response', y='Age', color='Response',
                    title="Âge vs Réponse",
                    color_discrete_map={0: '#e17055', 1: '#00b894'}
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Gender' in df.columns and 'Response' in df.columns:
                fig = px.bar(
                    df.groupby('Gender')['Response'].mean() * 100,
                    title="Taux par Genre",
                    color_discrete_sequence=['#00b894']
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.markdown("### Matrice de Corrélation")
            data_corr = df.copy()
            if 'id' in data_corr.columns:
                data_corr = data_corr.drop('id', axis=1)
            if 'dataset_type' in data_corr.columns:
                data_corr = data_corr.drop('dataset_type', axis=1)
            
            for col in data_corr.select_dtypes(include=['object']).columns:
                try:
                    data_corr[col] = LabelEncoder().fit_transform(data_corr[col])
                except:
                    data_corr = data_corr.drop(col, axis=1)
            
            corr = data_corr.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale='Teal', zmid=0
            ))
            fig.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Aucune donnée")

# ============================================================================
# PAGE PERFORMANCE - NOUVEAU DESIGN
# ============================================================================

def show_performance_page(df, model_type):
    """Affiche la page de performance"""
    
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;' class='slide-in'>
        <div style='font-size: 60px;'>📈</div>
        <h1 class='title-gradient'>Performance du Modèle</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        if st.button("🚀 Entraîner", type="primary"):
            with st.spinner('⏳ Entraînement en cours...'):
                data_processed, cat_cols, num_cols = preprocess_data(df)
                
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                    le_dict[col] = le
                
                X = data_processed.drop('Response', axis=1)
                y = data_processed['Response']
                
                model, scaler, X_test, y_test, y_pred, y_proba, metrics, cv_scores = train_ensemble_model(X, y, model_type)
                
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_proba'] = y_proba
                st.session_state['metrics'] = metrics
                st.session_state['cv_scores'] = cv_scores
                st.session_state['X'] = X
            
            st.success("✅ Entraînement terminé!")
        
        if 'metrics' in st.session_state:
            m = st.session_state['metrics']
            cv = st.session_state['cv_scores']
            
            # Métriques
            st.markdown("### 🎯 Métriques")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            cols[1].metric("Precision", f"{m['precision']*100:.1f}%")
            cols[2].metric("Recall", f"{m['recall']*100:.1f}%")
            cols[3].metric("F1-Score", f"{m['f1']*100:.1f}%")
            cols[4].metric("ROC-AUC", f"{m['roc_auc']*100:.1f}%")
            
            st.markdown("### 📊 Visualisations")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_confusion_matrix_plot(st.session_state['y_test'], st.session_state['y_pred'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_roc_curve_plot(st.session_state['y_test'], st.session_state['y_proba'])
                st.plotly_chart(fig, use_container_width=True)
            
            fig = create_precision_recall_plot(st.session_state['y_test'], st.session_state['y_proba'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔍 Importance")
            fig = create_feature_importance_plot(st.session_state['model'], st.session_state['X'].columns)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Aucune donnée")

# ============================================================================
# PAGE GUIDE - NOUVEAU DESIGN
# ============================================================================

def show_guide_page():
    """Affiche la page guide"""
    
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;' class='slide-in'>
        <div style='font-size: 60px;'>❓</div>
        <h1 class='title-gradient'>Guide & Documentation</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques
    with st.expander("📊 Métriques d'Évaluation", expanded=True):
        st.markdown("""
        - **Accuracy**: % de prédictions correctes
        - **Precision**: Qualité des prédictions positives
        - **Recall**: % de positifs trouvés
        - **F1-Score**: Moyenne Precision/Recall
        - **ROC-AUC**: Qualité globale du modèle
        """)
    
    # Variables
    with st.expander("📝 Description des Variables"):
        st.markdown("""
        | Variable | Description |
        |----------|-------------|
        | Gender | Genre (Male/Female) |
        | Age | Âge du client |
        | Driving_License | Permis conduire |
        | Region_Code | Code région |
        | Previously_Insured | Déjà assuré |
        | Vehicle_Age | Âge véhicule |
        | Vehicle_Damage | Dommages véhicule |
        | Annual_Premium | Prime annuelle |
        | Policy_Sales_Channel | Canal vente |
        | Vintage | Jours de relation |
        """)
    
    # Tips
    with st.expander("💡 Recommandations"):
        st.markdown("""
        1. Ajustez le seuil selon vos objectifs
        2. Considérez le coût faux positif vs faux négatif
        3. Retraînez régulièrement le modèle
        4. Utilisez la cross-validation
        """)

# ============================================================================
# FOOTER
# ============================================================================

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #636e72;'>
        <p>🚗 <b>TP2 IIA S6 2025-2026</b> - Modèles ML d'Ensemble</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    df, model_type, page = create_sidebar()
    
    if page == "🏠 Accueil":
        show_homepage()
    elif page == "🔮 Prédiction":
        show_prediction_page(df, model_type)
    elif page == "📊 Analyse":
        show_analysis_page(df)
    elif page == "📈 Performance":
        show_performance_page(df, model_type)
    elif page == "❓ Guide":
        show_guide_page()
    
    show_footer()

if __name__ == "__main__":
    main()
