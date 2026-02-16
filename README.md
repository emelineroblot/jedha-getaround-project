# 🚗 GetAround - Projet Data Science

## 📊 Contexte

**GetAround** est le Airbnb des voitures. Vous pouvez louer des voitures à n'importe qui pour quelques heures ou quelques jours !

### Problématique

Les retards au checkout génèrent des frictions pour le prochain conducteur. Ce projet vise à :
- Analyser les retards et leur impact
- Déterminer un seuil optimal de délai minimum entre deux locations
- Créer un modèle ML pour optimiser les prix de location

---

## 🎯 Objectifs du projet

### 1. 📊 Dashboard Streamlit
Analyse interactive des retards avec :
- Statistiques et visualisations des retards
- Simulateur de seuil minimum
- Recommandations basées sur les données

### 2. 🤖 API de prédiction de prix
API REST avec endpoint `/predict` pour prédire les prix de location optimaux

### 3. 📖 Documentation API
Documentation complète accessible via `/docs`

---

## 📂 Structure du projet

```
getaround_project/
├── data/                              # Données (non versionnées)
│   ├── get_around_delay_analysis.xlsx
│   └── get_around_pricing_project.csv
├── notebooks/                         # Analyses exploratoires
│   ├── 01_EDA_delays.ipynb
│   └── 02_ML_pricing.ipynb
├── dashboard/                         # Application Streamlit
│   ├── app.py
│   └── requirements.txt
├── api/                              # API FastAPI
│   ├── main.py
│   ├── model.pkl
│   └── requirements.txt
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔗 Démos en ligne

> **À compléter après déploiement**

- 📊 **Dashboard Streamlit** : [URL à venir]
- 🤖 **API** : [URL à venir]
- 📖 **Documentation API** : [URL]/docs

---

## 🚀 Installation locale

### Prérequis
- Python 3.9+
- pip

### Installation

```bash
# Cloner le repository
git clone <url-du-repo>
cd getaround_project

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer le dashboard localement

```bash
cd dashboard
streamlit run app.py
```

### Lancer l'API localement

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📈 Résultats

> **À compléter après analyse**

### Insights clés
- [À venir]

### Performance du modèle ML
- **R² Score** : [À venir]
- **RMSE** : [À venir]

---

## 🛠️ Technologies utilisées

- **Data Analysis** : Python, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning** : Scikit-learn
- **Dashboard** : Streamlit, Plotly
- **API** : FastAPI, Uvicorn
- **Déploiement** : Streamlit Cloud, Hugging Face Spaces

---

## 👤 Auteur

**Emeline** - Projet Jedha Formation

---

## 📝 Licence

Ce projet est réalisé dans le cadre de la formation Jedha.
