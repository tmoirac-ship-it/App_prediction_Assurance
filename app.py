"""
🏥 Application Streamlit - Prédiction de Réponse à l'Assurance Véhicule
TP2 IIA S6 - Modèles ML d'Ensemble

Auteur: HLDX - Henri Ledoux SAME
Date: 2025-2026
=================================================================
Interface Streamlit pour le déploiement de modèles ML d'ensemble
Version améliorée avec design moderne et interactif

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
    page_title="🏥 Prédiction Assurance Véhicule - TP2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## 🏥 Prédiction de Réponse à l'Assurance Véhicule
        
        Application de Machine Learning pour prédire si un client 
        est susceptible d'être intéressé par une assurance véhicule.
        
        Développé par HLDX - Henri Ledoux SAME
        Version: TP2 IIA S6 2025-2026
        """
    }
)

# ============================================================================
# STYLE CSS PERSONNALISÉ
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Style général */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers principaux */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Couleurs personnalisées - Titre gradient */
    .title-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .metric-card-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Dark mode card */
    .dark-card {
        background: #161b22;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Animation fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Divider styling */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Theme sombre */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #667eea !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Cards */
    div.stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    /* Messages */
    .stSuccess {
        background-color: #161b22;
        border-left: 4px solid #238636;
    }
    .stError {
        background-color: #161b22;
        border-left: 4px solid #da3633;
    }
    .stInfo {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
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
            st.error("Format de fichier non supporté. Utilisez CSV ou Excel.")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
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
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Définition des modèles
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Logistic Regression
    lr = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        C=0.5
    )
    
    # AdaBoost
    ada = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    
    if model_type == 'voting':
        # Voting Classifier (Soft Voting)
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',
            n_jobs=-1
        )
    elif model_type == 'stacking':
        # Stacking Classifier
        ensemble = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5
        )
    else:
        ensemble = rf
    
    # Entraînement
    ensemble.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
    }
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='f1')
    
    return ensemble, scaler, X_test_scaled, y_test, y_pred, y_proba, metrics, cv_scores

def create_feature_importance_plot(model, feature_names):
    """Créer un graphique d'importance des features avec Plotly"""
    # Extraire l'importance des features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_estimators_'):
        importances = model.named_estimators_['rf'].feature_importances_
    else:
        return None
    
    # Créer le DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Créer le graphique avec Plotly
    fig = px.bar(
        importance_df, 
        x='importance', 
        y='feature',
        orientation='h',
        title="🔍 Importance des Features dans la Prédiction",
        color='importance',
        color_continuous_scale='Viridis',
        text='importance'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Importance",
        yaxis_title="Features"
    )
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    """Créer une matrice de confusion avec Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit: Non', 'Prédit: Oui'],
        y=['Réel: Non', 'Réel: Oui'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=False
    ))
    
    fig.update_layout(
        title="📊 Matrice de Confusion",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_roc_curve_plot(y_true, y_proba):
    """Créer la courbe ROC avec Plotly"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Courbe ROC (AUC = {roc_auc:.3f})',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Aléatoire',
        line=dict(color='#eb3349', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="🎯 Courbe ROC",
        xaxis_title="Taux de Faux Positifs",
        yaxis_title="Taux de Vrais Positifs",
        template="plotly_white",
        height=400,
        legend=dict(x=0.7, y=0.1)
    )
    
    return fig

def create_precision_recall_plot(y_true, y_proba):
    """Créer la courbe Precision-Recall avec Plotly"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'Courbe PR (AP = {avg_precision:.3f})',
        line=dict(color='#38ef7d', width=3)
    ))
    
    fig.update_layout(
        title="📈 Courbe Precision-Recall",
        xaxis_title="Recall (Rappel)",
        yaxis_title="Precision",
        template="plotly_white",
        height=400,
        legend=dict(x=0.7, y=1)
    )
    
    return fig

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

def create_sidebar():
    """Crée la barre latérale de configuration"""
    
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: #667eea; margin: 0;'>⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Choix de la source de données
    st.sidebar.markdown("### 📁 Source de Données")
    data_source = st.sidebar.radio(
        "Sélectionner la source:",
        ["Dataset par défaut (merged_dataset.csv)", "Importer un fichier"]
    )
    
    df = None
    
    if data_source == "Importer un fichier":
        uploaded_file = st.sidebar.file_uploader(
            "Télécharger un fichier CSV ou Excel",
            type=['csv', 'xlsx']
        )
        if uploaded_file:
            df = load_data_from_file(uploaded_file)
            if df is not None:
                st.sidebar.success(f"Fichier chargé: {uploaded_file.name}")
    else:
        df = load_default_data()
        st.sidebar.info("Dataset merged_dataset.csv chargé par défaut")
    
    st.sidebar.markdown("---")
    
    # Configuration du modèle
    st.sidebar.markdown("### 🤖 Paramètres du Modèle")
    
    model_type = st.sidebar.selectbox(
        "Type de Modèle d'Ensemble",
        ['voting', 'stacking', 'random_forest'],
        format_func=lambda x: {
            'voting': '🗳️ Voting Classifier',
            'stacking': '🧱 Stacking Classifier',
            'random_forest': '🌲 Random Forest'
        }[x],
        help="Voting combine les prédictions, Stacking utilise un méta-modèle"
    )
    
    test_size = st.sidebar.slider("Taille du Test (%)", 10, 40, 20) / 100
    
    st.sidebar.markdown("---")
    
    # Navigation principale
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Aller à:",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse Exploratoire", "📈 Performance du Modèle", "🎓 Guide"]
    )
    
    st.sidebar.markdown("---")
    
    # Informations sur le projet
    st.sidebar.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); border-radius: 10px; padding: 15px; border: 1px solid #667eea;'>
        <h4 style='color: #667eea; margin: 0 0 10px 0;'>📋 À propos</h4>
        <p style='color: #c9d1d9; font-size: 12px; margin: 0;'>
            <b>TP2 IIA S6 2025-2026</b><br>
            Modèles ML d'Ensemble<br>
            Développé par HLDX - Henri Ledoux SAME
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return df, model_type, test_size, page

# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

def show_homepage():
    """Affiche la page d'accueil"""
    
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;' class='fade-in'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>
            <span class='title-gradient'>🏥 Prédiction Assurance Véhicule</span>
        </h1>
        <p style='font-size: 1.2rem; color: #888;'>
            Intelligence Artificielle pour identifier les clients susceptibles d'être intéressés par une assurance véhicule
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='dark-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>📊</h3>
            <h4>Analyse Interactive</h4>
            <p style='color: #888;'>Visualisez vos données en temps réel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='dark-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>🤖</h3>
            <h4>ML d'Ensemble</h4>
            <p style='color: #888;'>Random Forest, Gradient Boosting, Voting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='dark-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>📈</h3>
            <h4>Métriques Détaillées</h4>
            <p style='color: #888;'>Évaluation complète des performances</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='dark-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>🔮</h3>
            <h4>Prédiction</h4>
            <p style='color: #888;'>Testez avec vos propres données</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistiques rapides
    st.markdown("### 📈 Aperçu des Données")
    
    try:
        df = load_default_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Clients", len(df))
        with col2:
            if 'Response' in df.columns:
                positives = (df['Response'] == 1).sum()
                st.metric("✅ Intéressés", positives)
            else:
                st.metric("✅ Intéressés", "N/A")
        with col3:
            if 'Response' in df.columns:
                rate = (df['Response'] == 1).mean() * 100
                st.metric("📈 Taux d'Intérêt", f"{rate:.1f}%")
            else:
                st.metric("📈 Taux d'Intérêt", "N/A")
        with col4:
            st.metric("📋 Nombre de Features", df.shape[1] - 2)  # Exclude id and dataset_type
        
        # Aperçu des données
        st.markdown("### 👀 Aperçu des Données")
        st.dataframe(df.head(10), width='stretch')
        
    except Exception as e:
        st.warning(f"Impossible de charger les données: {e}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    <div class='dark-card fade-in'>
        <h2>🚀 Comment utiliser</h2>
        <ol style='font-size: 1.1rem; line-height: 2; color: #c9d1d9;'>
            <li><strong>Configurez les paramètres</strong> dans la barre latérale (modèle ML)</li>
            <li><strong>Allez dans "Prédiction"</strong> pour obtenir des prédictions</li>
            <li><strong>Analysez les performances</strong> dans la section "Performance du Modèle"</li>
            <li><strong>Explorez les données</strong> dans "Analyse Exploratoire"</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE PRÉDICTION
# ============================================================================

def show_prediction_page(df, model_type):
    """Affiche la page de prédiction"""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;' class='fade-in'>
        <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>
            <span class='title-gradient'>🔮 Prédiction pour un Nouveau Client</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Vérifier si le modèle est déjà entraîné dans la session
        model_key = f"model_{model_type}"
        
        if model_key not in st.session_state:
            with st.spinner('🔄 Entraînement du modèle en cours...'):
                data_processed, cat_cols, num_cols = preprocess_data(df)
                
                # Encoder les variables catégorielles
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                    le_dict[col] = le
                
                X = data_processed.drop('Response', axis=1)
                y = data_processed['Response']
                
                model, scaler, X_test, y_test, y_pred, y_proba, metrics, cv_scores = train_ensemble_model(X, y, model_type=model_type)
                
                # Stocker dans session_state
                st.session_state[model_key] = {
                    'model': model,
                    'scaler': scaler,
                    'le_dict': le_dict,
                    'X': X
                }
            st.success("✅ Modèle entraîné avec succès!")
        else:
            # Récupérer le modèle depuis session_state
            model_data = st.session_state[model_key]
            model = model_data['model']
            scaler = model_data['scaler']
            le_dict = model_data['le_dict']
            X = model_data['X']
            st.info("📌 Modèle chargé depuis le cache")
        
        # Formulaire de saisie
        st.markdown("### 📝 Informations du Client")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Démographie")
            gender = st.selectbox("Genre", ['Male', 'Female'], help="Genre du client")
            age = st.number_input("Âge (18-100)", min_value=18, max_value=100, value=35, 
                                help="Âge du client en années")
            driving_license = st.selectbox("Permis de Conduire", [0, 1], 
                                          format_func=lambda x: "Oui" if x == 1 else "Non",
                                          help="Le client possède-t-il un permis de conduire?")
            region_code = st.number_input("Code Région", min_value=0, max_value=60, value=28,
                                         help="Code de la région du client")
            
            st.markdown("#### 🚗 Véhicule")
            vehicle_age = st.selectbox("Âge du Véhicule", ['< 1 Year', '1-2 Year', '> 2 Years'],
                                      help="Ancienneté du véhicule")
            vehicle_damage = st.selectbox("Dommages au Véhicule", ['Yes', 'No'],
                                          help="Le véhicule a-t-il eu des dommages?")
        
        with col2:
            st.markdown("#### 🛡️ Assurances")
            previously_insured = st.selectbox("Déjà Assuré", [0, 1],
                                            format_func=lambda x: "Oui" if x == 1 else "Non",
                                            help="Le client a-t-il déjà une assurance véhicule?")
            
            st.markdown("#### 💰 Financier")
            annual_premium = st.number_input("Prime Annuelle (€)", value=30000, step=1000,
                                           help="Montant de la prime d'assurance annuelle")
            
            st.markdown("#### 📞 Contact & Campagne")
            policy_sales_channel = st.selectbox("Canal de Vente", 
                                                [26, 152, 124, 160, 156, 122, 13, 14],
                                                help="Canal de distribution utilisé")
            vintage = st.number_input("Ancienneté (jours)", min_value=0, max_value=300, value=200,
                                     help="Nombre de jours depuis que le client est en contact avec la société")
        
        st.markdown("---")
        
        # Bouton de prédiction
        if st.button("🔮 Obtenir la Prédiction", type="primary", use_container_width=True):
            # Créer le vecteur de caractéristiques
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Driving_License': [driving_license],
                'Region_Code': [region_code],
                'Previously_Insured': [previously_insured],
                'Vehicle_Age': [vehicle_age],
                'Vehicle_Damage': [vehicle_damage],
                'Annual_Premium': [annual_premium],
                'Policy_Sales_Channel': [policy_sales_channel],
                'Vintage': [vintage]
            })
            
            # Encoder les variables catégorielles
            for col in input_data.columns:
                if col in le_dict:
                    try:
                        input_data[col] = le_dict[col].transform([input_data[col].iloc[0]])[0]
                    except:
                        input_data[col] = 0
            
            # Standardiser
            input_scaled = scaler.transform(input_data)
            
            # Prédire
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            # Afficher le résultat
            st.markdown("---")
            st.markdown("### 🎯 Résultat de la Prédiction")
            
            col1, col2 = st.columns(2)
            
            if prediction == 1:
                with col1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                border-radius: 15px; padding: 30px; text-align: center; color: white;'>
                        <h1 style='margin: 0;'>✅ Intéressé!</h1>
                        <p style='font-size: 1.2rem;'>Le client est susceptible d'être intéressé par l'assurance véhicule</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Probabilité d'intérêt", f"{proba[1]*100:.1f}%")
                
                st.markdown("""
                <div class='success-box'>
                    <b>Recommandation:</b> Ce client est un bon candidat pour une offre d'assurance véhicule. 
                    Priorisez ce contact dans votre stratégie de ciblage.
                </div>
                """, unsafe_allow_html=True)
            else:
                with col1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                                border-radius: 15px; padding: 30px; text-align: center; color: white;'>
                        <h1 style='margin: 0;'>❌ Non Intéressé</h1>
                        <p style='font-size: 1.2rem;'>Le client a une faible probabilité d'intérêt</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Probabilité de désintérêt", f"{proba[0]*100:.1f}%")
                
                st.markdown("""
                <div class='warning-box'>
                    <b>Recommandation:</b> Ce client a une faible probabilité d'intérêt. 
                    Vous pouvez considérer d'allouer vos ressources à d'autres prospects.
                </div>
                """, unsafe_allow_html=True)
            
            # Facteurs d'influence
            st.markdown("---")
            st.markdown("### 🔍 Facteurs d'Influence")
            
            fig = create_feature_importance_plot(model, X.columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Veuillez charger un jeu de données dans la barre latérale.")

# ============================================================================
# PAGE ANALYSE EXPLORATOIRE
# ============================================================================

def show_analysis_page(df):
    """Affiche la page d'analyse exploratoire"""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;' class='fade-in'>
        <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>
            <span class='title-gradient'>📊 Analyse Exploratoire des Données</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Tabs pour différentes analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistiques", "🎯 Variable Cible", "👥 Démographie", "📊 Corrélations"])
        
        with tab1:
            st.markdown("### 📈 Statistiques Descriptives")
            
            # Statistiques numériques
            st.markdown("#### Variables Numériques")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if 'id' in numeric_df.columns:
                numeric_df = numeric_df.drop('id', axis=1)
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        with tab2:
            st.markdown("### 🎯 Analyse de la Variable Cible")
            
            if 'Response' in df.columns:
                # Graphique interactif avec Plotly
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Non Intéressé (0)', 'Intéressé (1)'],
                        y=[(df['Response'] == 0).sum(), (df['Response'] == 1).sum()],
                        marker_color=['#eb3349', '#11998e'],
                        text=[f"{(df['Response'] == 0).mean()*100:.1f}%", f"{(df['Response'] == 1).mean()*100:.1f}%"],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Distribution des Réponses",
                    xaxis_title="Classe",
                    yaxis_title="Nombre de clients",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total des Clients", len(df))
                with col2:
                    st.metric("Taux d'Intérêt", f"{(df['Response'] == 1).mean()*100:.1f}%")
        
        with tab3:
            st.markdown("### 👥 Analyse Démographique")
            
            # Analyse par âge avec Plotly
            if 'Age' in df.columns and 'Response' in df.columns:
                fig = px.histogram(
                    df, 
                    x='Age', 
                    color='Response',
                    title="Distribution par Âge selon la Réponse",
                    color_discrete_map={1: '#11998e', 0: '#eb3349'},
                    barmode='overlay'
                )
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse par genre
            if 'Gender' in df.columns and 'Response' in df.columns:
                gender_response = df.groupby('Gender')['Response'].mean() * 100
                
                fig = px.bar(
                    gender_response, 
                    x=gender_response.index,
                    y=gender_response.values,
                    title="Taux d'Intérêt par Genre",
                    color=gender_response.values,
                    color_continuous_scale='Viridis',
                    text=gender_response.values
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse par âge du véhicule
            if 'Vehicle_Age' in df.columns and 'Response' in df.columns:
                vehicle_age_response = df.groupby('Vehicle_Age')['Response'].mean() * 100
                
                fig = px.bar(
                    vehicle_age_response, 
                    x=vehicle_age_response.index,
                    y=vehicle_age_response.values,
                    title="Taux d'Intérêt par Âge du Véhicule",
                    color=vehicle_age_response.values,
                    color_continuous_scale='Viridis',
                    text=vehicle_age_response.values
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 📊 Matrice de Corrélation")
            
            # Préparer les données pour la corrélation
            data_corr = df.copy()
            
            # Supprimer les colonnes non numériques
            if 'id' in data_corr.columns:
                data_corr = data_corr.drop('id', axis=1)
            if 'dataset_type' in data_corr.columns:
                data_corr = data_corr.drop('dataset_type', axis=1)
            
            # Encoder les variables catégorielles pour la corrélation
            for col in data_corr.select_dtypes(include=['object']).columns:
                try:
                    data_corr[col] = LabelEncoder().fit_transform(data_corr[col])
                except:
                    data_corr = data_corr.drop(col, axis=1)
            
            # Calculer la corrélation
            corr_matrix = data_corr.corr()
            
            # Afficher la heatmap avec Plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8}
            ))
            
            fig.update_layout(
                title="Matrice de Corrélation des Variables",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Veuillez charger un jeu de données.")

# ============================================================================
# PAGE PERFORMANCE DU MODÈLE
# ============================================================================

def show_performance_page(df, model_type, test_size):
    """Affiche la page de performance du modèle"""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;' class='fade-in'>
        <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>
            <span class='title-gradient'>📈 Performance du Modèle</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Entraînement
        if st.button("🚀 Entraîner le Modèle", type="primary"):
            with st.spinner('🔄 Entraînement en cours...'):
                # Prétraitement
                data_processed, cat_cols, num_cols = preprocess_data(df)
                
                # Encoder
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                    le_dict[col] = le
                
                X = data_processed.drop('Response', axis=1)
                y = data_processed['Response']
                
                # Entraîner
                model, scaler, X_test, y_test, y_pred, y_proba, metrics, cv_scores = train_ensemble_model(
                    X, y, model_type=model_type
                )
                
                # Stocker dans session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_proba'] = y_proba
                st.session_state['metrics'] = metrics
                st.session_state['cv_scores'] = cv_scores
                st.session_state['X'] = X
            
            st.success("✅ Modèle entraîné avec succès!")
        
        # Afficher les résultats si disponibles
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            cv_scores = st.session_state['cv_scores']
            
            # Métriques principales
            st.markdown("### 🎯 Métriques de Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            with col2:
                st.metric("Precision", f"{metrics['precision']*100:.1f}%")
            with col3:
                st.metric("Recall", f"{metrics['recall']*100:.1f}%")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
            
            # Métriques supplémentaires
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ROC-AUC", f"{metrics['roc_auc']*100:.1f}%")
            with col2:
                st.metric("Cross-Val F1", f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
            
            st.markdown("---")
            
            # Visualisations
            st.markdown("### 📊 Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_confusion_matrix_plot(st.session_state['y_test'], st.session_state['y_pred'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_roc_curve_plot(st.session_state['y_test'], st.session_state['y_proba'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Precision-Recall Curve
            fig = create_precision_recall_plot(st.session_state['y_test'], st.session_state['y_proba'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.markdown("### 🔍 Importance des Features")
            fig = create_feature_importance_plot(
                st.session_state['model'], 
                st.session_state['X'].columns
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Veuillez charger un jeu de données.")

# ============================================================================
# PAGE GUIDE
# ============================================================================

def show_guide_page():
    """Affiche la page guide"""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;' class='fade-in'>
        <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>
            <span class='title-gradient'>🎓 Guide d'Interprétation</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques
    st.markdown("### 📊 Comprendre les Métriques")
    
    with st.expander("🎯 Accuracy (Précision Globale)", expanded=True):
        st.markdown("""
        **Définition:** Proportion de prédictions correctes parmi toutes les prédictions.
        
        - > 90%: Excellent
        - 80-90%: Bon
        - 70-80%: Acceptable
        - < 70%: À améliorer
        """)
    
    with st.expander("📈 Precision (Précision)"):
        st.markdown("""
        **Définition:** Proportion de prédictions positives qui sont correctes.
        
        Haute précision = Few false positives
        Utile quand le coût d'un faux positif est élevé
        """)
    
    with st.expander("📉 Recall (Rappel)"):
        st.markdown("""
        **Définition:** Proportion de positifs réels qui sont correctement identifiés.
        
        Haut recall = Few missed positives
        Utile quand le coût d'un faux négatif est élevé
        """)
    
    with st.expander("⚖️ F1-Score"):
        st.markdown("""
        **Définition:** Moyenne harmonique de Precision et Recall.
        
        Donne une vision équilibrée quand les classes sont déséquilibrées
        """)
    
    st.markdown("---")
    
    # Guide des features
    st.markdown("### 📝 Guide des Variables")
    
    with st.expander("👤 Démographie"):
        st.markdown("""
        - **Gender**: Genre du client (Male/Female)
        - **Age**: Âge du client en années
        - **Driving_License**: Permis de conduire (0/1)
        - **Region_Code**: Code de la région géographique
        """)
    
    with st.expander("🚗 Véhicule"):
        st.markdown("""
        - **Vehicle_Age**: Ancienneté du véhicule (< 1 Year, 1-2 Year, > 2 Years)
        - **Vehicle_Damage**: Antécédents de dommages au véhicule (Yes/No)
        """)
    
    with st.expander("🛡️ Assurance"):
        st.markdown("""
        - **Previously_Insured**: Si le client a déjà une assurance véhicule (0/1)
        - **Annual_Premium**: Montant de la prime d'assurance annuelle
        """)
    
    with st.expander("📞 Marketing"):
        st.markdown("""
        - **Policy_Sales_Channel**: Canal de distribution utilisé
        - **Vintage**: Nombre de jours depuis le premier contact
        """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Recommandations")
    
    st.markdown("""
    1. **Seuils de décision:** Ajuster selon les objectifs бизнеса
    2. **Coûts et bénéfices:** Considérer le coût de contact vs le gain potentiel
    3. **Mise à jour du modèle:** Retraîner régulièrement avec de nouvelles données
    4. **Validation:** Utiliser la cross-validation pour des estimations plus robustes
    """)

# ============================================================================
# PIED DE PAGE
# ============================================================================

def show_footer():
    """Affiche le pied de page"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>🏥 <strong>TP2 IIA S6 2025-2026</strong> - Modèles ML d'Ensemble</p>
        <p style='font-size: 0.9rem;'>Développé par HLDX - Henri Ledoux SAME avec Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # Créer la sidebar
    df, model_type, test_size, page = create_sidebar()
    
    # Affichage selon la page sélectionnée
    if page == "🏠 Accueil":
        show_homepage()
    
    elif page == "🔮 Prédiction":
        show_prediction_page(df, model_type)
    
    elif page == "📊 Analyse Exploratoire":
        show_analysis_page(df)
    
    elif page == "📈 Performance du Modèle":
        show_performance_page(df, model_type, test_size)
    
    elif page == "🎓 Guide":
        show_guide_page()
    
    show_footer()


if __name__ == "__main__":
    main()
