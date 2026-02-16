---
title: GetAround Pricing API
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🚗 GetAround Pricing API

API de Machine Learning pour prédire les prix optimaux de location de voitures.

## 🤖 À propos du modèle

- **Algorithme** : Random Forest Regressor
- **Performance** :
  - R² Score : 0.73 (73% de variance expliquée)
  - RMSE : 16.88€
  - MAPE : 14.84%
- **Dataset** : 4,843 locations de voitures
- **Features** : 56 caractéristiques (puissance moteur, kilométrage, équipements, etc.)

## 📖 Documentation

Une fois l'API déployée, accédez à :
- **Page d'accueil** : `/` - Interface HTML élégante avec toutes les infos
- **Swagger UI** : `/docs` - Documentation interactive pour tester l'API
- **Health check** : `/health` - Vérifier le statut

## 🔗 Endpoints disponibles

### POST /predict
Prédit le prix d'un ou plusieurs véhicules

**Input** :
```json
{
  "input": [
    [3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1, ...]
  ]
}
```

**Output** :
```json
{
  "prediction": [138.29]
}
```

### GET /health
Vérifie le statut de l'API

### GET /model-info
Retourne les informations détaillées du modèle ML

### GET /features
Liste des 56 features attendues

## 🚀 Utilisation

### Python
```python
import requests

# URL de votre Space (à remplacer)
API_URL = "https://YOUR-USERNAME-getaround-pricing-api.hf.space"

# Faire une prédiction
response = requests.post(
    f"{API_URL}/predict",
    json={
        "input": [
            [3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1] + [0]*46
        ]
    }
)

print(response.json())
# {'prediction': [138.29]}
```

### cURL
```bash
curl -X POST "https://YOUR-USERNAME-getaround-pricing-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"input": [[3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]}'
```

## 🎯 Format des features

Les 56 features doivent être dans l'ordre suivant :
1. **Unnamed: 0** - Index
2. **mileage** - Kilométrage (en km)
3. **engine_power** - Puissance moteur (en CV)
4. **private_parking_available** - Parking privé (0/1)
5. **has_gps** - GPS (0/1)
6. **has_air_conditioning** - Climatisation (0/1)
7. **automatic_car** - Boîte automatique (0/1)
8. **has_getaround_connect** - Service Connect (0/1)
9. **has_speed_regulator** - Régulateur de vitesse (0/1)
10. **winter_tires** - Pneus hiver (0/1)
11-56. **Features encodées** - Marque, couleur, type (one-hot)

Pour la liste complète : `GET /features`

## 📊 Projet

**Contexte** : Projet Jedha Bootcamp - Bloc Deployment

**Objectif** : Créer une API ML pour aider GetAround à optimiser les prix de location.

**Technologies** :
- FastAPI
- Scikit-learn
- Pandas, NumPy
- Uvicorn
- Docker

## 👤 Auteur

Projet réalisé dans le cadre du bootcamp Jedha Data Science.

## 📄 Licence

MIT License
