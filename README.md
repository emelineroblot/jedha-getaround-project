# 🚗 GetAround — Analyse des retards & optimisation des prix

Projet de fin de formation Jedha (bloc Deployment). Deux livrables complémentaires à partir des
données GetAround : un **tableau de bord d'aide à la décision** sur le délai minimum entre deux
locations, et une **API de prédiction de prix** mise en production.

## 🔗 Démos en ligne

| Livrable | URL |
|---|---|
| 📊 Dashboard Streamlit | <https://emeliner-jedha-getaround-streamlit.hf.space/> |
| 🤖 API de pricing | <https://emeliner-jedha-getaround-project.hf.space/> |
| 📖 Documentation de l'API | <https://emeliner-jedha-getaround-project.hf.space/docs> |
| 🧪 Documentation interactive | <https://emeliner-jedha-getaround-project.hf.space/swagger> |

> Les Spaces Hugging Face gratuits se mettent en veille après une période d'inactivité. Le premier
> chargement peut demander une trentaine de secondes, le temps du réveil.

---

## 🎯 Le problème

Quand un conducteur rend une voiture en retard, le conducteur suivant attend — voire annule.
GetAround envisage d'imposer un **délai minimum entre deux locations** : la voiture n'apparaît plus
dans les résultats de recherche si le créneau demandé est trop proche d'une réservation existante.

Cela réduit la friction, mais coûte du chiffre d'affaires aux propriétaires. Le Product Manager
doit trancher deux questions : **quel seuil**, et **sur quel périmètre** (tous les véhicules,
ou seulement les Connect ?).

---

## 📊 Résultats

### Les quatre questions du Product Manager

| Question | Réponse |
|---|---|
| **1. Part du revenu affectée** | **1,31 %** à 30 min (≈ 34 k€), **3,13 %** à 2 h (≈ 81 k€), avant report de la demande |
| **2. Locations impactées** | 279 à 30 min, 666 à 2 h — sur 21 310 locations. Seules **1 841 locations (8,6 %)** peuvent être concernées, puisque ce sont les seules à en suivre une autre |
| **3. Fréquence et impact des retards** | **57,5 %** des checkouts renseignés sont en retard (médiane 53 min). Mais seuls **218 conducteurs (1,0 % du parc)** subissent réellement un retard, avec une attente médiane de 27 min |
| **4. Cas problématiques résolus** | **53 %** à 30 min, **83 %** à 2 h, **90 %** à 3 h — avec une saturation nette au-delà de 2 h |

### Recommandation

**Un seuil court de 30 à 60 minutes, différencié par type de checkin** — à condition de valoriser
un incident évité à au moins 150–250 €. En dessous, la mesure coûte plus qu'elle ne rapporte.

Le dashboard expose ce compromis sous forme de deux curseurs (coût d'un incident, taux de report
de la demande) plutôt que d'imposer une formule : **la vraie question posée au PM n'est pas
« quel seuil ? » mais « combien vaut un client qui attend ? »**. Le seuil s'en déduit.

**Périmètre** : commencer par **Mobile**, où la friction est concentrée (149 des 218 cas), et non
par Connect comme le suggère l'intuition. Connect est le flux le plus fiable (43 % de retards
contre 61 %, retards deux à trois fois plus courts) : y déployer la mesure en premier sacrifierait
du chiffre d'affaires là où le problème est le moins présent.

### Un lien chiffré entre retard et perte de revenu

Un conducteur qui subit le retard du précédent annule **5,1 points plus souvent** (17,0 % contre
11,8 %). Corrélation observée, causalité non établie — mais l'ordre de grandeur permet de valoriser
un incident évité.

### Modèle de pricing

| Métrique | Valeur |
|---|---|
| Algorithme | Gradient Boosting Regressor (`Pipeline` scikit-learn) |
| R² (test) | **0,756** |
| R² (validation croisée, 5 folds) | **0,770 ± 0,032** |
| RMSE | **16,33 €** |
| MAE | **10,49 €** |
| MAPE | **13,69 %** |

Pour un prix médian de 119 €/jour, une erreur absolue moyenne de 10,49 € est exploitable comme
**recommandation** de prix. La puissance moteur et le kilométrage expliquent à eux seuls **72 %**
du prix.

---

## 📂 Structure du projet

```
jedha-getaround-project/
├── data/                        # Datasets (téléchargés automatiquement, non versionnés)
│   └── README.md                #   schémas et URLs sources
├── notebooks/
│   ├── 01_EDA_delays.ipynb      # Analyse des retards, simulation, recommandation
│   └── 02_ML_pricing.ipynb      # Exploration, comparaison de modèles, évaluation
├── src/
│   └── train_model.py           # Entraînement reproductible -> deployment/api/model.pkl
├── deployment/
│   ├── api/                     # Space HF « API »      (Dockerfile, main.py, model.pkl)
│   └── dashboard/               # Space HF « Dashboard » (Dockerfile, app.py, analysis.py)
├── tests/                       # 50 tests pytest (API, analyse, dashboard)
├── requirements.txt             # Environnement de développement complet
└── README.md
```

### Un principe : une seule source de vérité

`deployment/dashboard/analysis.py` contient **toute** la logique d'analyse des retards, et il est
importé à l'identique par le notebook `01_EDA_delays.ipynb`. Le notebook et le dashboard ne peuvent
donc pas afficher des chiffres différents.

De même, `src/train_model.py` est importé par `02_ML_pricing.ipynb` : le notebook explore et
justifie, le script produit le modèle déployé. Aucune duplication, aucune divergence possible.

Les dossiers `deployment/api` et `deployment/dashboard` sont poussés tels quels vers Hugging Face :
**le code visible sur GitHub est exactement celui qui tourne en production.**

---

## 🚀 Installation

**Prérequis** : Python 3.11 ou supérieur (3.13 recommandé — la version utilisée en production).

```bash
git clone https://github.com/emelineroblot/jedha-getaround-project
cd jedha-getaround-project

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Aucun téléchargement manuel de données n'est nécessaire : tout est récupéré à la première
exécution.

### Lancer le dashboard

```bash
cd deployment/dashboard
streamlit run app.py            # http://localhost:8501
```

### Lancer l'API

```bash
cd deployment/api
uvicorn main:app --reload --port 8000
# http://localhost:8000/docs      documentation rédigée
# http://localhost:8000/swagger   documentation interactive
```

### Réentraîner le modèle

```bash
python src/train_model.py                  # entraînement complet + GridSearch + MLflow
python src/train_model.py --no-gridsearch  # version rapide

mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

### Lancer les tests

```bash
pytest tests -v
```

50 tests couvrent le contrat de l'API, la logique d'analyse (dont deux tests de non-régression sur
les erreurs de méthode corrigées) et l'exécution du dashboard.

---

## 🤖 Utiliser l'API

### `POST /predict`

Entrée : un objet JSON avec la clé `input`, contenant une liste de vecteurs de **59 valeurs**
dans l'ordre renvoyé par `GET /features`.

```bash
curl -i -H "Content-Type: application/json" -X POST \
  -d "{\"input\": [[140000, 135, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]}" \
  https://emeliner-jedha-getaround-project.hf.space/predict
```

```json
{"prediction": [139.75]}
```

```python
import requests

API = "https://emeliner-jedha-getaround-project.hf.space"

n = requests.get(f"{API}/features").json()["count"]
vector = [140000, 135] + [0] * (n - 2)

response = requests.post(f"{API}/predict", json={"input": [vector]})
print(response.json())        # {'prediction': [114.41]}
```

### `POST /predict/form` — plus confortable

Ordonner 59 valeurs à la main est pénible. Cet endpoint accepte les caractéristiques nommées et
fait l'encodage côté serveur :

```python
requests.post(f"{API}/predict/form", json={
    "mileage": 140000, "engine_power": 135,
    "model_key": "BMW", "fuel": "diesel", "car_type": "sedan",
    "has_gps": True, "has_getaround_connect": True,
}).json()
```

### Tous les endpoints

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/` | GET | Page d'accueil |
| `/docs` | GET | Documentation rédigée |
| `/swagger` | GET | Documentation interactive |
| `/predict` | POST | Prédiction (vecteur positionnel) |
| `/predict/form` | POST | Prédiction (features nommées) |
| `/health` | GET | État de l'API |
| `/model-info` | GET | Métriques et métadonnées du modèle |
| `/features` | GET | Liste ordonnée des features |
| `/version` | GET | Versions API / modèle / scikit-learn |

---

## ⚠️ Limites assumées

**Sur l'analyse des retards**

- 23 % des locations n'ont pas de donnée de checkout (essentiellement les annulations). Elles sont
  exclues du dénominateur du taux de retard, jamais comptées comme « à l'heure ».
- 112 locations consécutives ne peuvent pas être classées, faute de retard connu pour la location
  précédente : l'estimation de 218 cas problématiques est donc légèrement conservatrice.
- Les chiffres de blocage sont des **majorants** : ils comptent des locations observées qui
  n'auraient pas eu lieu, sans modéliser le report de la demande sur un autre créneau.
- Les deux jeux de données n'ont **aucune clé commune** (pas de `car_id` côté pricing) : les
  montants en euros reposent sur un prix moyen de 121,21 €/jour et une durée supposée d'un jour.

**Sur le modèle de pricing**

- L'écart train/test est de +0,13, soit un surapprentissage léger et assumé. La validation croisée
  (0,770 ± 0,032) confirme que la performance n'est pas un artefact du découpage.
- La dispersion des erreurs croît avec le prix : le modèle est moins fiable au-dessus de 200 €/jour.
- Aucune variable de contexte (ville, saison, jour de la semaine, tension de l'offre) n'est
  disponible, alors que ce sont des déterminants majeurs d'un prix de location.
- Le modèle prédit le prix **pratiqué**, pas le prix **optimal** : il reproduit les habitudes de
  tarification existantes. Une vraie optimisation demanderait des données de conversion.

---

## 🛠️ Stack technique

| Domaine | Outils |
|---|---|
| Analyse | pandas, NumPy |
| Visualisation | Plotly (dashboard), Matplotlib / Seaborn (notebooks) |
| Machine Learning | scikit-learn, MLflow |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Tests | pytest, `streamlit.testing`, `fastapi.TestClient` |
| Déploiement | Docker, Hugging Face Spaces |

---

## 👤 Auteur

**Emeline Roblot** — Projet réalisé dans le cadre de la formation Jedha Data Science.

## 📝 Licence

MIT.
