"""
Entraînement reproductible du modèle de pricing GetAround.

Usage
-----
    python src/train_model.py                 # entraînement complet + sauvegarde
    python src/train_model.py --no-gridsearch # version rapide (~10 s)
    python src/train_model.py --no-mlflow     # sans tracking MLflow

Le script est autonome : il télécharge le dataset s'il est absent, entraîne,
compare plusieurs modèles, valide par validation croisée, puis écrit
`deployment/api/model.pkl`.

Choix de conception importants
------------------------------
* La colonne `Unnamed: 0` du CSV est l'index de ligne, PAS une caractéristique du
  véhicule. Elle est neutralisée par `index_col=0`. Un modèle qui s'en sert
  apprend l'ordre du fichier, ce qui ne se généralise à rien.
* Le préprocessing est encapsulé dans un `Pipeline` scikit-learn, sauvegardé tel
  quel. L'API ne réimplémente donc aucune étape de transformation.
* Les métriques sauvegardées sont TOUJOURS recalculées sur le modèle finalement
  retenu (après GridSearch), jamais héritées d'un modèle intermédiaire.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deployment" / "dashboard"))
import analysis  # noqa: E402  (dépend de sys.path ci-dessus)

RANDOM_STATE = 42
TARGET = "rental_price_per_day"
MODEL_OUTPUT = REPO_ROOT / "deployment" / "api" / "model.pkl"


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Charge le dataset pricing et applique un nettoyage minimal documenté."""
    df = analysis.load_pricing_data()
    report = {"n_raw": len(df)}

    # Doublons stricts
    n_before = len(df)
    df = df.drop_duplicates()
    report["n_duplicates_dropped"] = n_before - len(df)

    # Valeurs physiquement impossibles : un kilométrage négatif ou une puissance
    # nulle sont des erreurs de saisie, pas des véhicules réels.
    invalid = (df["mileage"] < 0) | (df["engine_power"] <= 0)
    report["n_invalid_dropped"] = int(invalid.sum())
    df = df[~invalid]

    report["n_clean"] = len(df)
    return df, report


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encoding des variables catégorielles.

    `drop_first=False` est volontaire : toutes les modalités restent explicites,
    ce qui rend l'endpoint `/features` de l'API auto-documenté et permet à un
    utilisateur de construire un payload pour n'importe quelle marque. La
    colinéarité induite est absorbée par la régularisation (Ridge) et sans effet
    sur les modèles à base d'arbres.
    """
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=categorical, drop_first=False)

    # Les booléens deviennent des entiers : le JSON de l'API transporte des 0/1.
    bool_cols = encoded.select_dtypes(include=["bool"]).columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)

    y = encoded[TARGET]
    X = encoded.drop(columns=[TARGET])
    return X, y


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def compute_metrics(model, X_train, y_train, X_test, y_test) -> dict:
    """Métriques train/test d'un modèle DÉJÀ entraîné."""
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "r2_train": float(r2_score(y_train, pred_train)),
        "r2_test": float(r2_score(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "mape_test": float(mean_absolute_percentage_error(y_test, pred_test) * 100),
    }


def candidate_models() -> dict[str, Pipeline]:
    """Modèles comparés.

    Le `StandardScaler` est indispensable aux modèles linéaires et neutre pour
    les arbres (invariants aux transformations monotones) : le garder dans tous
    les pipelines uniformise le format sauvegardé sans coût de performance.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE,
            )),
        ]),
        # La cible est strictement positive et asymétrique à droite : on teste
        # explicitement une modélisation en log, réflexe attendu sur un prix.
        "Gradient Boosting (log target)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", TransformedTargetRegressor(
                regressor=GradientBoostingRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05,
                    random_state=RANDOM_STATE,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            )),
        ]),
    }


GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 20],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
    },
}


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def main(run_gridsearch: bool = True, use_mlflow: bool = True) -> dict:
    print("=" * 78)
    print("Entraînement du modèle de pricing GetAround")
    print("=" * 78)

    df, clean_report = load_and_clean()
    print(f"\nDonnées : {clean_report['n_raw']} lignes brutes -> {clean_report['n_clean']} nettoyées "
          f"({clean_report['n_duplicates_dropped']} doublon(s), "
          f"{clean_report['n_invalid_dropped']} valeur(s) aberrante(s) retirée(s))")

    X, y = build_features(df)
    feature_names = X.columns.tolist()
    print(f"Features : {X.shape[1]} colonnes après encodage "
          f"(la colonne d'index 'Unnamed: 0' est exclue)")

    # Le pipeline est entraîné sur un tableau NumPy, et non sur un DataFrame :
    # l'API sert des vecteurs positionnels, ce contrat évite le UserWarning
    # « X does not have valid feature names » à chaque prédiction en production.
    X_values = X.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Split    : {len(X_train)} train / {len(X_test)} test")

    # --- Comparaison des candidats ---------------------------------------- #
    print("\n" + "-" * 78)
    print("Comparaison des modèles")
    print("-" * 78)
    results = {}
    for name, pipeline in candidate_models().items():
        pipeline.fit(X_train, y_train)
        metrics = compute_metrics(pipeline, X_train, y_train, X_test, y_test)
        results[name] = {"pipeline": pipeline, "metrics": metrics}
        print(f"{name:32s} R2_test={metrics['r2_test']:.4f}  "
              f"RMSE={metrics['rmse_test']:6.2f}  MAPE={metrics['mape_test']:5.2f}%  "
              f"(gap train-test={metrics['r2_train'] - metrics['r2_test']:+.3f})")

    best_name = max(results, key=lambda n: results[n]["metrics"]["r2_test"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\nMeilleur candidat : {best_name}")

    # --- Optimisation ------------------------------------------------------ #
    if run_gridsearch and best_name in GRIDS:
        print("\n" + "-" * 78)
        print(f"GridSearchCV sur {best_name}")
        print("-" * 78)
        search = GridSearchCV(
            candidate_models()[best_name],
            GRIDS[best_name],
            cv=5, scoring="r2", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        tuned_metrics = compute_metrics(tuned, X_train, y_train, X_test, y_test)
        print(f"Meilleurs paramètres : {search.best_params_}")
        print(f"R2_test optimisé     : {tuned_metrics['r2_test']:.4f} "
              f"(vs {results[best_name]['metrics']['r2_test']:.4f})")

        if tuned_metrics["r2_test"] > results[best_name]["metrics"]["r2_test"]:
            best_pipeline = tuned
            print("-> modèle optimisé retenu")
        else:
            print("-> modèle de base conservé (plus simple, performance équivalente)")

    # --- Métriques FINALES, recalculées sur le modèle réellement retenu ---- #
    # Ce recalcul est ce qui garantit que le pickle ne publie jamais les
    # métriques d'un autre modèle que celui qu'il contient.
    final_metrics = compute_metrics(best_pipeline, X_train, y_train, X_test, y_test)

    # --- Validation croisée sur l'ensemble des données --------------------- #
    print("\n" + "-" * 78)
    print("Validation croisée (5 folds, sur l'ensemble du jeu de données)")
    print("-" * 78)
    cv_scores = cross_val_score(
        best_pipeline, X_values, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2", n_jobs=-1,
    )
    final_metrics["r2_cv_mean"] = float(cv_scores.mean())
    final_metrics["r2_cv_std"] = float(cv_scores.std())
    print(f"R2 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  "
          f"(folds : {', '.join(f'{s:.3f}' for s in cv_scores)})")

    gap = final_metrics["r2_train"] - final_metrics["r2_test"]
    print(f"\nÉcart train-test : {gap:+.3f} "
          f"({'surapprentissage à assumer' if gap > 0.1 else 'généralisation saine'})")

    # --- Importance des features ------------------------------------------- #
    importances = extract_importances(best_pipeline, feature_names)
    if importances is not None:
        print("\nTop 10 des features :")
        for feature, weight in importances.head(10).items():
            print(f"   {feature:34s} {weight:6.2%}")

    # --- MLflow ------------------------------------------------------------ #
    if use_mlflow:
        log_to_mlflow(best_name, best_pipeline, final_metrics, clean_report, len(feature_names))

    # --- Sauvegarde -------------------------------------------------------- #
    package = {
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "model_name": best_name,
        "target_name": TARGET,
        "metrics": final_metrics,
        "cleaning_report": clean_report,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_importances": (importances.to_dict() if importances is not None else {}),
    }
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODEL_OUTPUT)

    print("\n" + "=" * 78)
    print(f"Modèle sauvegardé : {MODEL_OUTPUT.relative_to(REPO_ROOT)}")
    print(f"   Algorithme : {best_name}")
    print(f"   Features   : {len(feature_names)}")
    print(f"   R2 test    : {final_metrics['r2_test']:.4f}")
    print(f"   RMSE       : {final_metrics['rmse_test']:.2f} EUR")
    print(f"   MAPE       : {final_metrics['mape_test']:.2f} %")
    print(f"   sklearn    : {sklearn.__version__} (à épingler dans requirements.txt)")
    print("=" * 78)

    return package


def extract_importances(pipeline: Pipeline, columns: list[str]) -> pd.Series | None:
    """Importances du modèle final, quel que soit son emballage."""
    model = pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if not hasattr(model, "feature_importances_"):
        return None
    return (
        pd.Series(model.feature_importances_, index=columns)
        .sort_values(ascending=False)
    )


def log_to_mlflow(name, pipeline, metrics, clean_report, n_features) -> None:
    """Tracking MLflow local (base sqlite ./mlflow.db, non versionnée).

    Le backend « fichier » (./mlruns) est déprécié depuis février 2026 : on
    utilise directement une base sqlite, qui donne accès au registre de modèles.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("\n[MLflow] paquet absent - tracking ignoré.")
        return

    try:
        mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("getaround-pricing")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model": name,
                "n_features": n_features,
                "random_state": RANDOM_STATE,
                "rows_clean": clean_report["n_clean"],
                "sklearn_version": sklearn.__version__,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name="model")
        print("\n[MLflow] run enregistré dans ./mlflow.db "
              "(visualisation : mlflow ui --backend-store-uri sqlite:///mlflow.db)")
    except Exception as exc:  # le tracking ne doit jamais casser l'entraînement
        print(f"\n[MLflow] tracking ignoré ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gridsearch", action="store_true",
                        help="ignorer l'optimisation des hyperparamètres")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="désactiver le tracking MLflow")
    args = parser.parse_args()
    main(run_gridsearch=not args.no_gridsearch, use_mlflow=not args.no_mlflow)
