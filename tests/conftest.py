"""Configuration commune des tests : rend l'API et le module d'analyse importables."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deployment" / "api"))
sys.path.insert(0, str(REPO_ROOT / "deployment" / "dashboard"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Racine du dépôt, pour les tests qui lisent la documentation."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def api_module():
    """Le module FastAPI, modèle chargé."""
    import main

    if not main.model_loaded:
        pytest.skip(
            "model.pkl absent : lancer `python src/train_model.py` avant les tests."
        )
    return main


@pytest.fixture(scope="session")
def client(api_module):
    """Client de test synchrone sur l'application FastAPI."""
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def n_features(api_module) -> int:
    return len(api_module.feature_names)


@pytest.fixture(scope="session")
def delays():
    """Jeu de données des retards, enrichi (téléchargé au premier appel)."""
    import analysis

    return analysis.enrich(analysis.load_delay_data())
