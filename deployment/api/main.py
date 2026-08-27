"""
GetAround Pricing API — prédiction du prix optimal de location.

Endpoints
---------
GET  /            page d'accueil HTML
GET  /docs        documentation rédigée (titre h1, exigence de l'énoncé)
GET  /swagger     documentation interactive Swagger UI
POST /predict     prédiction à partir d'un vecteur de features ordonné
POST /predict/form prédiction à partir de features nommées (confort)
GET  /health      état de l'API
GET  /model-info  métriques et métadonnées du modèle
GET  /features    liste ordonnée des features attendues
GET  /version     versions de l'API, du modèle et de scikit-learn
"""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Chemin ABSOLU : l'API doit démarrer quel que soit le répertoire courant
# (`uvicorn main:app` depuis deployment/api, ou `uvicorn deployment.api.main:app`
# depuis la racine du dépôt).
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

API_VERSION = "2.0.0"
API_TITLE = "GetAround Pricing API"

# --------------------------------------------------------------------------- #
# État du modèle
# --------------------------------------------------------------------------- #
model_package: dict[str, Any] = {}
pipeline = None
feature_names: list[str] = []
model_metrics: dict[str, float] = {}
model_loaded = False


def load_model() -> bool:
    """Charge le modèle sérialisé. Retourne False plutôt que de lever."""
    global model_package, pipeline, feature_names, model_metrics, model_loaded

    if not MODEL_PATH.exists():
        print(f"[modele] introuvable : {MODEL_PATH}")
        model_loaded = False
        return False

    try:
        model_package = joblib.load(MODEL_PATH)
        pipeline = model_package["pipeline"]
        feature_names = list(model_package["feature_names"])
        model_metrics = dict(model_package.get("metrics", {}))
        model_loaded = True
        print(f"[modele] {model_package.get('model_name')} charge "
              f"({len(feature_names)} features, R2={model_metrics.get('r2_test', float('nan')):.4f})")
        return True
    except Exception as exc:
        print(f"[modele] erreur de chargement : {type(exc).__name__}: {exc}")
        model_loaded = False
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application (remplace les `on_event` dépréciés)."""
    load_model()
    yield
    print("[api] arret")


def _metrics_summary() -> str:
    """Résumé textuel des performances, dérivé du modèle réellement chargé.

    Rien n'est écrit en dur : après un réentraînement, la documentation suit.
    """
    if not model_loaded:
        return "Modèle non chargé."
    return (
        f"- **Algorithme** : {model_package.get('model_name', 'inconnu')}\n"
        f"- **R²** (test) : {model_metrics.get('r2_test', float('nan')):.4f}\n"
        f"- **RMSE** : {model_metrics.get('rmse_test', float('nan')):.2f} €\n"
        f"- **MAPE** : {model_metrics.get('mape_test', float('nan')):.2f} %\n"
        f"- **Features** : {len(feature_names)}\n"
    )


load_model()

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "API de Machine Learning pour prédire le prix optimal de location "
        "d'un véhicule GetAround.\n\n## Modèle\n\n" + _metrics_summary() +
        "\n\nDocumentation rédigée : [/docs](/docs) — "
        "documentation interactive : [/swagger](/swagger)"
    ),
    # L'énoncé exige une page de documentation à /docs comportant un titre h1.
    # Swagger UI rend son titre dans un <h2>, on la déplace donc sur /swagger et
    # on sert à /docs une page rédigée qui satisfait explicitement l'exigence.
    docs_url="/swagger",
    redoc_url="/redoc",
    lifespan=lifespan,
    license_info={"name": "MIT"},
)


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #
class PredictionInput(BaseModel):
    """Vecteur(s) de features, dans l'ordre exact renvoyé par `GET /features`."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"input": [[140000, 135] + [0] * 57]}}
    )

    input: list[list[float]] = Field(
        ...,
        description="Liste de vecteurs de features. Un vecteur = un véhicule.",
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("La liste 'input' ne peut pas être vide.")
        lengths = {len(row) for row in value}
        if len(lengths) > 1:
            raise ValueError(
                f"Tous les vecteurs doivent avoir la même longueur ; reçu {sorted(lengths)}."
            )
        return value


class PredictionOutput(BaseModel):
    """Prix prédits, en euros par jour."""

    model_config = ConfigDict(json_schema_extra={"example": {"prediction": [138.29]}})

    prediction: list[float] = Field(..., description="Prix prédits (€/jour).")


class FormPredictionInput(BaseModel):
    """Features nommées : les champs absents prennent leur valeur par défaut."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mileage": 140000,
                "engine_power": 135,
                "model_key": "BMW",
                "fuel": "diesel",
                "paint_color": "black",
                "car_type": "sedan",
                "has_gps": True,
                "has_getaround_connect": True,
            }
        },
        extra="allow",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str
    features_count: int
    api_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    features_count: int
    metrics: dict[str, float]
    feature_names: list[str]
    trained_at: str | None = None
    sklearn_version: str | None = None
    target: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_model() -> None:
    if not model_loaded or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé : l'API n'est pas opérationnelle.",
        )


def _predict_matrix(matrix: np.ndarray) -> list[float]:
    """Prédit et post-traite : arrondi à 2 décimales, prix jamais négatif."""
    predictions = pipeline.predict(matrix)
    return [max(round(float(value), 2), 0.0) for value in predictions]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def read_root() -> HTMLResponse:
    """Page d'accueil : état de l'API, métriques et liens utiles."""
    status_class = "ok" if model_loaded else "error"
    status_label = "Opérationnel" if model_loaded else "Modèle non chargé"
    return HTMLResponse(_render_page(
        title=f"{API_TITLE}",
        body=f"""
        <p><strong>Version :</strong> {API_VERSION}
           &nbsp;·&nbsp; <span class="status {status_class}">{status_label}</span></p>

        <h2>Performances du modèle</h2>
        <div class="metrics">
          <div class="metric"><div class="metric-value">{model_metrics.get('r2_test', 0):.3f}</div>
            <div class="metric-label">R² (test)</div></div>
          <div class="metric"><div class="metric-value">{model_metrics.get('rmse_test', 0):.2f} €</div>
            <div class="metric-label">RMSE</div></div>
          <div class="metric"><div class="metric-value">{model_metrics.get('mape_test', 0):.2f} %</div>
            <div class="metric-label">MAPE</div></div>
          <div class="metric"><div class="metric-value">{len(feature_names)}</div>
            <div class="metric-label">Features</div></div>
        </div>

        <h2>Où aller ensuite</h2>
        <ul>
          <li><a href="/docs">/docs</a> — documentation rédigée de tous les endpoints</li>
          <li><a href="/swagger">/swagger</a> — documentation interactive (test en direct)</li>
          <li><a href="/health">/health</a> — état de l'API</li>
          <li><a href="/features">/features</a> — liste ordonnée des features attendues</li>
        </ul>
        """,
    ))


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def documentation() -> HTMLResponse:
    """Documentation rédigée de l'API.

    Page servie à `/docs` conformément à l'énoncé : elle comporte un titre `h1`
    et décrit, pour chaque endpoint, son nom, sa méthode HTTP, l'entrée requise
    et la sortie attendue.
    """
    n_features = len(feature_names) or 59
    example_vector = "[" + ", ".join(["140000", "135"] + ["0"] * (n_features - 2)) + "]"
    short_vector = "[140000, 135, 0, 0, ... ]"

    return HTMLResponse(_render_page(
        title="GetAround Pricing API — Documentation",
        heading_level=1,
        body=f"""
        <p>Cette API expose un modèle de Machine Learning entraîné à prédire le
        <strong>prix de location optimal d'un véhicule, en euros par jour</strong>.
        Toutes les réponses sont au format JSON, sauf les pages HTML
        <code>/</code> et <code>/docs</code>.</p>

        <h2>Modèle utilisé</h2>
        <div class="metrics">
          <div class="metric"><div class="metric-value">{model_package.get('model_name', 'n/a')}</div>
            <div class="metric-label">Algorithme</div></div>
          <div class="metric"><div class="metric-value">{model_metrics.get('r2_test', 0):.3f}</div>
            <div class="metric-label">R² (test)</div></div>
          <div class="metric"><div class="metric-value">{model_metrics.get('rmse_test', 0):.2f} €</div>
            <div class="metric-label">RMSE</div></div>
          <div class="metric"><div class="metric-value">{len(feature_names)}</div>
            <div class="metric-label">Features</div></div>
        </div>

        <h2>POST /predict</h2>
        <div class="endpoint">
          <span class="method post">POST</span><strong>/predict</strong>
          <p><strong>Entrée requise</strong> — un objet JSON avec la clé <code>input</code>,
          contenant une liste de vecteurs. Chaque vecteur comporte exactement
          <strong>{n_features} valeurs numériques</strong>, dans l'ordre renvoyé par
          <a href="/features">GET /features</a>.</p>
<pre><code>{{
  "input": [{short_vector}]
}}</code></pre>
          <p><strong>Sortie attendue</strong> — un objet JSON avec la clé
          <code>prediction</code> : la liste des prix prédits, dans le même ordre
          que les vecteurs fournis.</p>
<pre><code>{{
  "prediction": [138.29]
}}</code></pre>
          <p><strong>Codes de retour</strong> : <code>200</code> succès ·
          <code>400</code> nombre de features incorrect ·
          <code>422</code> corps de requête invalide ·
          <code>503</code> modèle non chargé.</p>
        </div>

        <h2>POST /predict/form</h2>
        <div class="endpoint">
          <span class="method post">POST</span><strong>/predict/form</strong>
          <p><strong>Entrée requise</strong> — les caractéristiques du véhicule
          <em>nommées</em>, ce qui évite d'ordonner {n_features} valeurs à la main.
          Tout champ omis prend la valeur 0 (ou la modalité absente).</p>
<pre><code>{{
  "mileage": 140000,
  "engine_power": 135,
  "model_key": "BMW",
  "fuel": "diesel",
  "car_type": "sedan",
  "has_gps": true
}}</code></pre>
          <p><strong>Sortie attendue</strong> — identique à <code>/predict</code> :</p>
<pre><code>{{
  "prediction": [141.87]
}}</code></pre>
        </div>

        <h2>GET /health</h2>
        <div class="endpoint">
          <span class="method get">GET</span><strong>/health</strong>
          <p><strong>Entrée requise</strong> — aucune.</p>
          <p><strong>Sortie attendue</strong> — état de l'API et du modèle :</p>
<pre><code>{{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "{model_package.get('model_name', 'n/a')}",
  "features_count": {len(feature_names)},
  "api_version": "{API_VERSION}",
  "timestamp": "2026-08-27T10:00:00+00:00"
}}</code></pre>
          <p>Retourne <code>503</code> si le modèle n'a pas pu être chargé.</p>
        </div>

        <h2>GET /model-info</h2>
        <div class="endpoint">
          <span class="method get">GET</span><strong>/model-info</strong>
          <p><strong>Entrée requise</strong> — aucune.</p>
          <p><strong>Sortie attendue</strong> — nom de l'algorithme, métriques de
          performance (R², RMSE, MAE, MAPE, R² en validation croisée), date
          d'entraînement, version de scikit-learn et liste complète des features.</p>
        </div>

        <h2>GET /features</h2>
        <div class="endpoint">
          <span class="method get">GET</span><strong>/features</strong>
          <p><strong>Entrée requise</strong> — aucune.</p>
          <p><strong>Sortie attendue</strong> — la liste ordonnée des features
          attendues par <code>/predict</code>, et leur nombre :</p>
<pre><code>{{
  "features": ["mileage", "engine_power", "private_parking_available", "..."],
  "count": {len(feature_names)}
}}</code></pre>
        </div>

        <h2>GET /version</h2>
        <div class="endpoint">
          <span class="method get">GET</span><strong>/version</strong>
          <p><strong>Entrée requise</strong> — aucune.</p>
          <p><strong>Sortie attendue</strong> — versions de l'API, du modèle,
          de scikit-learn et de Python.</p>
        </div>

        <h2>Exemples d'appel</h2>
        <p>En cURL :</p>
<pre><code>curl -i -H "Content-Type: application/json" -X POST \\
  -d '{{"input": [{example_vector}]}}' \\
  https://emeliner-jedha-getaround-project.hf.space/predict</code></pre>
        <p>En Python :</p>
<pre><code>import requests

response = requests.post(
    "https://emeliner-jedha-getaround-project.hf.space/predict",
    json={{"input": [{short_vector}]}},
)
print(response.json())   # {{'prediction': [138.29]}}</code></pre>

        <p>Pour tester les endpoints directement depuis le navigateur, utilisez la
        documentation interactive : <a href="/swagger">/swagger</a>.</p>
        """,
    ))


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Vérifie que l'API et le modèle sont opérationnels."""
    _require_model()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_package.get("model_name", "inconnu"),
        features_count=len(feature_names),
        api_version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(data: PredictionInput) -> PredictionOutput:
    """Prédit le prix de location d'un ou plusieurs véhicules.

    Chaque vecteur doit contenir les features dans l'ordre exact renvoyé par
    `GET /features`. Le prix retourné est exprimé en euros par jour.
    """
    _require_model()

    matrix = np.asarray(data.input, dtype=float)
    if matrix.shape[1] != len(feature_names):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Nombre de features incorrect : {len(feature_names)} attendues, "
                f"{matrix.shape[1]} reçues. Voir GET /features pour la liste ordonnée."
            ),
        )

    return PredictionOutput(prediction=_predict_matrix(matrix))


@app.post("/predict/form", response_model=PredictionOutput, tags=["Prediction"])
async def predict_form(data: FormPredictionInput) -> PredictionOutput:
    """Prédit un prix à partir de features **nommées**.

    Confort d'usage : l'encodage one-hot est fait côté serveur, il n'y a donc
    pas besoin d'ordonner manuellement les dizaines de colonnes attendues par
    `/predict`. Les champs absents valent 0.
    """
    _require_model()

    payload = data.model_dump()
    vector = dict.fromkeys(feature_names, 0.0)
    unknown: list[str] = []

    for key, value in payload.items():
        if value is None:
            continue
        if key in vector:                       # feature numérique ou booléenne
            vector[key] = float(value)
        elif f"{key}_{value}" in vector:        # modalité one-hot, ex. model_key_BMW
            vector[f"{key}_{value}"] = 1.0
        else:
            unknown.append(f"{key}={value}")

    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Champs ou modalités inconnus : {', '.join(unknown)}. "
                "Voir GET /features pour les valeurs acceptées."
            ),
        )

    matrix = np.asarray([[vector[name] for name in feature_names]], dtype=float)
    return PredictionOutput(prediction=_predict_matrix(matrix))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info() -> ModelInfoResponse:
    """Métriques et métadonnées du modèle chargé."""
    _require_model()
    return ModelInfoResponse(
        model_name=model_package.get("model_name", "inconnu"),
        features_count=len(feature_names),
        metrics=model_metrics,
        feature_names=feature_names,
        trained_at=model_package.get("trained_at"),
        sklearn_version=model_package.get("sklearn_version"),
        target=model_package.get("target_name"),
    )


@app.get("/features", tags=["Model"])
async def get_features() -> dict[str, Any]:
    """Liste ordonnée des features attendues par `/predict`."""
    _require_model()
    return {
        "features": feature_names,
        "count": len(feature_names),
        "description": (
            f"Les {len(feature_names)} features attendues par POST /predict, "
            "dans cet ordre exact."
        ),
    }


@app.get("/version", tags=["Info"])
async def get_version() -> dict[str, Any]:
    """Versions de l'API, du modèle et de l'environnement d'exécution."""
    return {
        "api_version": API_VERSION,
        "model_name": model_package.get("model_name") if model_loaded else None,
        "model_trained_at": model_package.get("trained_at") if model_loaded else None,
        "sklearn_version": model_package.get("sklearn_version") if model_loaded else None,
        "python_version": platform.python_version(),
    }


# --------------------------------------------------------------------------- #
# Gestion des erreurs
# --------------------------------------------------------------------------- #
@app.exception_handler(404)
async def not_found_handler(request, exc) -> JSONResponse:
    """Réponse 404 propre.

    Un handler d'exception DOIT retourner un objet Response : renvoyer un dict
    fait lever `TypeError: 'dict' object is not callable` au niveau ASGI, et
    transforme chaque 404 en erreur serveur.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Endpoint non trouvé",
            "detail": "Consultez /docs pour la liste des endpoints disponibles.",
            "path": str(request.url.path),
        },
    )


# --------------------------------------------------------------------------- #
# Rendu HTML partagé
# --------------------------------------------------------------------------- #
def _render_page(title: str, body: str, heading_level: int = 1) -> str:
    """Gabarit HTML commun aux pages `/` et `/docs`."""
    tag = f"h{heading_level}"
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
           max-width: 900px; margin: 0 auto; padding: 32px 20px;
           background: #f5f6fa; color: #22252a; line-height: 1.6; }}
    .container {{ background: #fff; padding: 32px 40px; border-radius: 12px;
                 box-shadow: 0 2px 16px rgba(0,0,0,.08); }}
    h1 {{ color: #4b3fbb; border-bottom: 3px solid #4b3fbb; padding-bottom: 12px;
         margin-top: 0; }}
    h2 {{ color: #5f4bb6; margin-top: 36px; }}
    code, pre {{ font-family: 'SFMono-Regular', Consolas, monospace; }}
    pre {{ background: #1e2129; color: #e6e6e6; padding: 14px 16px;
          border-radius: 6px; overflow-x: auto; font-size: 13px; }}
    pre code {{ background: none; color: inherit; padding: 0; }}
    code {{ background: #eceefb; padding: 2px 6px; border-radius: 4px; font-size: 90%; }}
    a {{ color: #4b3fbb; font-weight: 600; }}
    .endpoint {{ background: #fafbff; padding: 16px 20px; margin: 16px 0;
                border-radius: 8px; border-left: 4px solid #4b3fbb; }}
    .method {{ display: inline-block; padding: 3px 10px; border-radius: 4px;
              font-weight: 700; color: #fff; margin-right: 10px; font-size: 12px; }}
    .get {{ background: #3b82f6; }} .post {{ background: #10b981; }}
    .status {{ display: inline-block; padding: 4px 14px; border-radius: 20px;
              font-size: 13px; font-weight: 700; }}
    .status.ok {{ background: #d1fae5; color: #065f46; }}
    .status.error {{ background: #fee2e2; color: #991b1b; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
               gap: 14px; margin: 20px 0; }}
    .metric {{ background: #eceefb; padding: 16px; border-radius: 8px; text-align: center; }}
    .metric-value {{ font-size: 22px; font-weight: 700; color: #4b3fbb; }}
    .metric-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <{tag}>{title}</{tag}>
    {body}
    <p style="text-align:center;margin-top:40px;color:#9aa0aa;font-size:13px;">
      Projet Jedha — GetAround Analysis
    </p>
  </div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    # `reload` exige une chaîne d'import, pas l'objet application : passer `app`
    # directement ferait désactiver le rechargement silencieusement.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
