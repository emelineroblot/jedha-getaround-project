# 📓 Guide d'utilisation des Notebooks

## 🚀 Lancer Jupyter Lab

### Option 1 : Depuis le terminal
```bash
# Activer l'environnement virtuel (si pas déjà activé)
cd C:\Users\Emeline\Documents\_DEV\_Projets_Jedha\getaround_project
.\venv\Scripts\activate

# Lancer Jupyter Lab
jupyter lab

# OU lancer Jupyter Notebook classique
jupyter notebook
```

### Option 2 : Commande rapide (depuis le dossier du projet)
```bash
cd notebooks
..\venv\Scripts\jupyter lab
```

## 📊 Notebooks disponibles

### 01_EDA_delays.ipynb
**Objectif** : Analyse exploratoire des retards

**Contenu** :
- Chargement et exploration des données
- Distribution des retards
- Analyse par type de checkin (Mobile, Connect, Paper)
- Impact sur les locations suivantes
- Simulation de différents seuils de délai minimum
- Graphiques interactifs
- Recommandations

**Durée estimée** : 2-3 heures

**Outputs** :
- Insights clés sur les retards
- Fichier CSV avec résultats de simulation des seuils
- Graphiques pour le dashboard

---

### 02_ML_pricing.ipynb (à créer)
**Objectif** : Modèle de prédiction de prix

**Contenu** :
- EDA des données de pricing
- Préparation des données
- Feature engineering
- Entraînement de modèles
- Évaluation et optimisation
- Sauvegarde du modèle

**Durée estimée** : 4-5 heures

---

## 💡 Conseils d'utilisation

### 1. Exécution des cellules
- **Shift + Enter** : Exécuter la cellule et passer à la suivante
- **Ctrl + Enter** : Exécuter la cellule sans changer de cellule
- **Alt + Enter** : Exécuter la cellule et en insérer une nouvelle en dessous

### 2. Ordre d'exécution
⚠️ **Important** : Exécutez les cellules dans l'ordre ! Les cellules dépendent des variables créées précédemment.

### 3. Redémarrer le kernel
Si vous rencontrez des erreurs :
- Menu : `Kernel` → `Restart Kernel and Clear All Outputs`
- Puis ré-exécuter toutes les cellules

### 4. Sauvegarder régulièrement
- **Ctrl + S** pour sauvegarder
- Jupyter sauvegarde automatiquement toutes les 2 minutes

---

## 📦 Dépendances nécessaires

Toutes les dépendances sont déjà installées dans l'environnement virtuel :
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- openpyxl (pour lire les fichiers Excel)

---

## 🐛 Troubleshooting

### Problème : Le kernel ne démarre pas
**Solution** :
```bash
python -m ipykernel install --user --name=getaround_venv
```

### Problème : Module introuvable
**Solution** :
```bash
.\venv\Scripts\pip install [nom_du_module]
```

### Problème : Graphiques Plotly ne s'affichent pas
**Solution** :
```bash
.\venv\Scripts\pip install plotly nbformat
# Puis redémarrer le kernel
```

---

## 📁 Structure des données

```
data/
├── get_around_delay_analysis.xlsx      # Données des retards
├── get_around_pricing_project.csv      # Données de pricing
└── threshold_simulation_results.csv    # Résultats (généré par le notebook)
```

---

Bon travail ! 🚀
