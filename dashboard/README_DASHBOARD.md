# 🎨 Dashboard Streamlit - GetAround

## 🚀 Lancer le dashboard localement

### Méthode 1 : Depuis le dossier dashboard
```bash
cd C:\Users\Emeline\Documents\_DEV\_Projets_Jedha\getaround_project\dashboard
..\venv\Scripts\streamlit run app.py
```

### Méthode 2 : Depuis la racine du projet
```bash
cd C:\Users\Emeline\Documents\_DEV\_Projets_Jedha\getaround_project
.\venv\Scripts\activate
streamlit run dashboard/app.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

---

## 📊 Fonctionnalités du dashboard

### 1. **Vue d'ensemble**
- Métriques clés (KPIs)
  - Total de locations
  - Pourcentage de retards
  - Retard moyen
  - Cas problématiques
- Distribution des retards (histogramme + pie chart)

### 2. **Analyse par type de checkin**
- Comparaison Mobile vs Connect
- Pourcentage de retards par type
- Retard moyen par type
- Tableau récapitulatif détaillé

### 3. **Simulateur interactif** ⭐
- **Slider** pour ajuster le seuil (0 à 720 minutes)
- **Sélecteur de périmètre** (Tous / Connect / Mobile)
- **Métriques en temps réel** :
  - Nombre de locations bloquées
  - Nombre de problèmes résolus
  - Impact estimé sur les revenus
- **Graphique Trade-off** : courbes locations bloquées vs problèmes résolus
- **Tableau comparatif** de différents seuils

### 4. **Recommandations**
- Observations clés basées sur les données
- Seuil optimal calculé automatiquement
- Stratégie de déploiement suggérée

### 5. **Données brutes**
- Aperçu des données (expandable)
- Statistiques descriptives

---

## ⚙️ Paramètres disponibles

### Sidebar
- **Filtres par type de checkin** : Mobile, Connect
- Les graphiques et métriques s'ajustent automatiquement

### Simulateur
- **Seuil** : 0 à 720 minutes (par pas de 30 min)
- **Périmètre** : Tous / Connect uniquement / Mobile uniquement

---

## 📁 Structure des fichiers

```
dashboard/
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
└── README_DASHBOARD.md     # Ce fichier
```

---

## 🎨 Captures d'écran

Le dashboard comprend :
- 📊 Graphiques interactifs (Plotly)
- 🎛️ Contrôles en temps réel
- 📈 Visualisations claires et colorées
- 💡 Recommandations basées sur les données

---

## 🐛 Troubleshooting

### Problème : Module introuvable
**Erreur :** `ModuleNotFoundError: No module named 'streamlit'`

**Solution :**
```bash
cd C:\Users\Emeline\Documents\_DEV\_Projets_Jedha\getaround_project
.\venv\Scripts\activate
pip install -r dashboard/requirements.txt
```

### Problème : Fichier de données introuvable
**Erreur :** `FileNotFoundError: get_around_delay_analysis.xlsx`

**Solution :**
Vérifiez que le fichier est bien dans `data/get_around_delay_analysis.xlsx`

### Problème : Le dashboard ne se rafraîchit pas
**Solution :**
- Appuyez sur `R` dans le terminal pour recharger
- Ou utilisez le bouton "Rerun" dans l'interface Streamlit

### Problème : Erreur openpyxl
**Solution :**
```bash
.\venv\Scripts\pip install openpyxl
```

---

## 🚀 Déploiement en production

### Sur Streamlit Cloud (gratuit)

1. **Pousser le code sur GitHub**
```bash
git add dashboard/
git commit -m "Add Streamlit dashboard"
git push origin main
```

2. **Déployer sur Streamlit Cloud**
- Aller sur [streamlit.io/cloud](https://streamlit.io/cloud)
- Se connecter avec GitHub
- Cliquer sur "New app"
- Sélectionner le repo et le fichier `dashboard/app.py`
- Cliquer sur "Deploy"

3. **Configuration**
- Streamlit détectera automatiquement `requirements.txt`
- Assurez-vous que le fichier de données est accessible (dans le repo ou via URL)

⚠️ **Important** : Si les données sont volumineuses (>100 MB), utilisez Git LFS ou hébergez-les ailleurs (S3, etc.)

---

## 📊 Performance

- **Temps de chargement** : ~2-3 secondes
- **Cache activé** : Les données sont mises en cache avec `@st.cache_data`
- **Réactivité** : Les graphiques se mettent à jour instantanément

---

## 🎯 Prochaines améliorations possibles

- [ ] Export PDF du rapport
- [ ] Téléchargement des données filtrées (CSV)
- [ ] Comparaison de plusieurs scénarios
- [ ] Prédictions avec ML (intégration du modèle de pricing)
- [ ] Authentification utilisateur
- [ ] Mode sombre/clair

---

Bon travail ! 🚀
