---
title: GetAround Pricing API
emoji: 🚗
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GetAround Pricing API

API de Machine Learning prédisant le prix de location journalier optimal d'un véhicule.

> Ce dossier est déployé tel quel sur Hugging Face Spaces. Le code source de référence vit dans
> le dépôt GitHub du projet : <https://github.com/emelineroblot/jedha-getaround-project>

## Endpoints

| Endpoint | Méthode | Entrée | Sortie |
|---|---|---|---|
| `/` | GET | — | Page d'accueil HTML |
| `/docs` | GET | — | Documentation rédigée (titre `h1`) |
| `/swagger` | GET | — | Documentation interactive Swagger UI |
| `/predict` | POST | `{"input": [[...59 valeurs...]]}` | `{"prediction": [139.75]}` |
| `/predict/form` | POST | Features nommées | `{"prediction": [124.68]}` |
| `/health` | GET | — | État de l'API et du modèle |
| `/model-info` | GET | — | Métriques et métadonnées du modèle |
| `/features` | GET | — | Liste ordonnée des 59 features |
| `/version` | GET | — | Versions API / modèle / scikit-learn |

## Exemple d'appel

```bash
curl -i -H "Content-Type: application/json" -X POST \
  -d "{\"input\": [[140000, 135, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]}" \
  https://emeliner-jedha-getaround-project.hf.space/predict
```

```python
import requests

API = "https://emeliner-jedha-getaround-project.hf.space"

# Option 1 — vecteur positionnel (contrat de l'énoncé)
n = requests.get(f"{API}/features").json()["count"]
vector = [140000, 135] + [0] * (n - 2)
print(requests.post(f"{API}/predict", json={"input": [vector]}).json())

# Option 2 — features nommées, plus confortable
print(requests.post(f"{API}/predict/form", json={
    "mileage": 140000, "engine_power": 135,
    "model_key": "BMW", "fuel": "diesel", "car_type": "sedan",
    "has_gps": True,
}).json())
```

## Modèle

Les métriques exactes du modèle chargé sont exposées par `GET /model-info` — la documentation ne
les duplique pas, afin qu'elles ne puissent jamais diverger après un réentraînement.

Algorithme : **Gradient Boosting Regressor** encapsulé avec sa standardisation dans un `Pipeline`
scikit-learn. Entraîné par `python src/train_model.py` dans le dépôt GitHub.

## Reproductibilité

`requirements.txt` épingle les versions **à l'identique de l'environnement d'entraînement**. Un
modèle scikit-learn sérialisé n'est pas garanti compatible entre versions : dés-épingler expose à
un échec de dépicklage au prochain rebuild du Space.

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# http://localhost:8000/docs
```
