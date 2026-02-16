"""
🚗 GetAround - API de Prédiction de Prix
API FastAPI pour prédire les prix optimaux de location de voitures
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ===== CONFIGURATION =====
MODEL_PATH = 'model.pkl'
API_VERSION = "1.0.0"
API_TITLE = "GetAround Pricing API"
API_DESCRIPTION = """
🚗 **GetAround Pricing API**

API de Machine Learning pour prédire les prix optimaux de location de voitures.

## Fonctionnalités

* **Prédiction de prix** : Obtenez le prix optimal basé sur les caractéristiques du véhicule
* **Documentation automatique** : Interface Swagger UI interactive
* **Health check** : Vérification du statut de l'API
* **Informations du modèle** : Métriques et features utilisées

## Modèle utilisé

- **Algorithme** : Random Forest Regressor
- **Performance** : R² = 0.73, MAPE = 14.84%
- **Features** : 56 caractéristiques (puissance, kilométrage, équipements, etc.)

## Auteur

Projet Jedha - GetAround Analysis
"""

# ===== INITIALISATION DE L'API =====
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact={
        "name": "GetAround Team",
        "url": "https://github.com/jedha-bootcamp",
    },
    license_info={
        "name": "MIT",
    },
)

# ===== CHARGEMENT DU MODÈLE =====
model_package = None
model = None
scaler = None
feature_names = []
model_metrics = {}

def load_model():
    """Charge le modèle au démarrage de l'API"""
    global model_package, model, scaler, feature_names, model_metrics

    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Fichier modèle introuvable : {MODEL_PATH}")
            return False

        model_package = joblib.load(MODEL_PATH)
        model = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        model_metrics = model_package.get('metrics', {})

        print("✅ Modèle chargé avec succès")
        print(f"   - Modèle : {model_package.get('model_name', 'Unknown')}")
        print(f"   - Features : {len(feature_names)}")
        print(f"   - R² : {model_metrics.get('r2_test', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return False

# Charger le modèle au démarrage
model_loaded = load_model()

# ===== MODÈLES PYDANTIC =====

class PredictionInput(BaseModel):
    """Format d'entrée pour la prédiction"""
    input: List[List[float]] = Field(
        ...,
        description="Liste de listes de features. Chaque liste interne représente un véhicule à prédire.",
        example=[[0, 140000, 120, 1, 1, 0, 0, 1, 1, 1] + [0]*46]  # Exemple simplifié
    )

    @validator('input')
    def validate_input(cls, v):
        if not v:
            raise ValueError("La liste d'input ne peut pas être vide")
        if not all(isinstance(item, list) for item in v):
            raise ValueError("Chaque élément doit être une liste")
        return v

    class Config:
        schema_extra = {
            "example": {
                "input": [
                    [3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                ]
            }
        }

class PredictionOutput(BaseModel):
    """Format de sortie pour la prédiction"""
    prediction: List[float] = Field(
        ...,
        description="Liste des prix prédits en euros par jour"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": [138.29]
            }
        }

class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    model_loaded: bool
    model_name: str
    features_count: int
    api_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Informations sur le modèle"""
    model_name: str
    features_count: int
    metrics: Dict[str, float]
    feature_names: List[str]

# ===== ENDPOINTS =====

@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def read_root():
    """
    Page d'accueil de l'API avec les liens principaux
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{API_TITLE}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #764ba2;
                margin-top: 30px;
            }}
            .endpoint {{
                background: #f8f9fa;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .method {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                color: white;
                margin-right: 10px;
            }}
            .get {{ background-color: #61affe; }}
            .post {{ background-color: #49cc90; }}
            a {{
                color: #667eea;
                text-decoration: none;
                font-weight: bold;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .status {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            .status.ok {{
                background: #d4edda;
                color: #155724;
            }}
            .status.error {{
                background: #f8d7da;
                color: #721c24;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .metric {{
                background: #e7f3ff;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 {API_TITLE}</h1>
            <p><strong>Version:</strong> {API_VERSION}</p>
            <p><strong>Statut:</strong> <span class="status {'ok' if model_loaded else 'error'}">{'✅ Opérationnel' if model_loaded else '❌ Modèle non chargé'}</span></p>

            <h2>📊 Métriques du Modèle</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{model_metrics.get('r2_test', 0):.2%}</div>
                    <div class="metric-label">R² Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{model_metrics.get('rmse_test', 0):.1f}€</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{model_metrics.get('mape_test', 0):.1f}%</div>
                    <div class="metric-label">MAPE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(feature_names)}</div>
                    <div class="metric-label">Features</div>
                </div>
            </div>

            <h2>🔗 Endpoints Disponibles</h2>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/predict</strong><br>
                Prédire le prix d'un ou plusieurs véhicules
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong><br>
                Vérifier le statut de l'API
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/model-info</strong><br>
                Obtenir les informations du modèle ML
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/features</strong><br>
                Liste des features attendues par le modèle
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/docs</strong><br>
                Documentation interactive Swagger UI
            </div>

            <h2>📖 Documentation</h2>
            <p>
                Pour une documentation interactive complète, visitez :
                <a href="/docs" target="_blank">📄 /docs (Swagger UI)</a>
            </p>

            <h2>💡 Exemple d'utilisation</h2>
            <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto;">
<code>import requests

# Faire une prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json={{
        "input": [[3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1, ...]]
    }}
)

print(response.json())
# {{"prediction": [138.29]}}</code>
            </pre>

            <p style="text-align: center; margin-top: 40px; color: #999;">
                Made with ❤️ for Jedha Bootcamp | GetAround Project
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Vérifie que l'API fonctionne correctement

    Returns:
        - status: "healthy" si tout va bien, "unhealthy" sinon
        - model_loaded: True si le modèle est chargé
        - model_name: Nom du modèle ML
        - features_count: Nombre de features
        - api_version: Version de l'API
        - timestamp: Date et heure actuelles
    """
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. L'API n'est pas opérationnelle."
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_package.get('model_name', 'Unknown'),
        features_count=len(feature_names),
        api_version=API_VERSION,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(data: PredictionInput):
    """
    Effectue des prédictions de prix pour un ou plusieurs véhicules

    **Input Format:**
    ```json
    {
        "input": [
            [feature1, feature2, ..., feature56],
            [feature1, feature2, ..., feature56]
        ]
    }
    ```

    Chaque liste interne doit contenir exactement **56 features** dans l'ordre suivant :
    - Unnamed: 0, mileage, engine_power, private_parking_available, has_gps,
      has_air_conditioning, automatic_car, has_getaround_connect,
      has_speed_regulator, winter_tires, + 46 features one-hot encodées

    **Output Format:**
    ```json
    {
        "prediction": [price1, price2, ...]
    }
    ```

    Les prix sont en euros par jour.
    """
    try:
        # Vérifier que le modèle est chargé
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé"
            )

        # Convertir en numpy array
        X = np.array(data.input)

        # Vérifier la shape
        if X.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Nombre de features incorrect. Attendu: {len(feature_names)}, Reçu: {X.shape[1]}"
            )

        # Standardiser
        X_scaled = scaler.transform(X)

        # Prédire
        predictions = model.predict(X_scaled)

        # Arrondir à 2 décimales et s'assurer que les prix sont positifs
        predictions = [max(round(float(p), 2), 0) for p in predictions]

        return PredictionOutput(prediction=predictions)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur de validation : {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Retourne les informations détaillées sur le modèle ML

    Returns:
        - model_name: Nom de l'algorithme
        - features_count: Nombre de features
        - metrics: Métriques de performance (R², RMSE, MAE, MAPE)
        - feature_names: Liste complète des features
    """
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé"
        )

    return ModelInfoResponse(
        model_name=model_package.get('model_name', 'Unknown'),
        features_count=len(feature_names),
        metrics=model_metrics,
        feature_names=feature_names
    )

@app.get("/features", tags=["Model"])
async def get_features():
    """
    Retourne la liste des features attendues par le modèle

    Utile pour construire les inputs de prédiction correctement.
    """
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé"
        )

    return {
        "features": feature_names,
        "count": len(feature_names),
        "description": "Liste des 56 features attendues dans l'ordre exact pour /predict"
    }

@app.get("/version", tags=["Info"])
async def get_version():
    """Retourne la version de l'API"""
    return {
        "api_version": API_VERSION,
        "model_version": model_package.get('model_name', 'Unknown') if model_loaded else None,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }

# ===== GESTION DES ERREURS =====

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handler pour les routes non trouvées"""
    return {
        "error": "Endpoint non trouvé",
        "detail": "Consultez /docs pour la liste des endpoints disponibles",
        "path": str(request.url)
    }

# ===== ÉVÉNEMENTS =====

@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'API"""
    print("="*80)
    print(f"🚀 {API_TITLE} v{API_VERSION}")
    print("="*80)
    if model_loaded:
        print("✅ API prête à recevoir des requêtes")
        print(f"📊 Modèle : {model_package.get('model_name', 'Unknown')}")
        print(f"🎯 R² Score : {model_metrics.get('r2_test', 'N/A')}")
    else:
        print("⚠️ ATTENTION : Modèle non chargé !")
    print("="*80)

@app.on_event("shutdown")
async def shutdown_event():
    """Événement à l'arrêt de l'API"""
    print("\n👋 Arrêt de l'API GetAround")

# ===== POINT D'ENTRÉE =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
