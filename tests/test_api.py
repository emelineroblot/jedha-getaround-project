"""Tests de l'API de pricing GetAround.

Contrat vérifié : celui de l'énoncé — `POST /predict` accepte `{"input": [[...]]}`
et retourne `{"prediction": [...]}`, et `/docs` sert une page de documentation
comportant un titre `h1`.
"""

import pytest


# --------------------------------------------------------------------------- #
# Contrat /predict imposé par l'énoncé
# --------------------------------------------------------------------------- #
def test_predict_retourne_la_cle_prediction(client, n_features):
    response = client.post("/predict", json={"input": [[0.0] * n_features]})

    assert response.status_code == 200
    payload = response.json()
    assert list(payload) == ["prediction"]
    assert isinstance(payload["prediction"], list)
    assert len(payload["prediction"]) == 1
    assert isinstance(payload["prediction"][0], (int, float))


def test_predict_accepte_plusieurs_vehicules(client, n_features):
    vehicles = [[140000, 135] + [0.0] * (n_features - 2),
                [50000, 200] + [0.0] * (n_features - 2)]
    response = client.post("/predict", json={"input": vehicles})

    assert response.status_code == 200
    assert len(response.json()["prediction"]) == 2


def test_predict_retourne_un_prix_positif_et_plausible(client, n_features):
    vehicle = [140000, 135] + [0.0] * (n_features - 2)
    price = client.post("/predict", json={"input": [vehicle]}).json()["prediction"][0]

    assert price > 0
    assert 10 <= price <= 500, f"prix hors de l'intervalle observé : {price}"


def test_predict_est_deterministe(client, n_features):
    payload = {"input": [[140000, 135] + [0.0] * (n_features - 2)]}
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()

    assert first == second


# --------------------------------------------------------------------------- #
# Gestion des erreurs (bonus de l'énoncé)
# --------------------------------------------------------------------------- #
def test_predict_refuse_un_mauvais_nombre_de_features(client):
    response = client.post("/predict", json={"input": [[1.0, 2.0, 3.0]]})

    assert response.status_code == 400
    assert "features" in response.json()["detail"].lower()


def test_predict_refuse_une_liste_vide(client):
    assert client.post("/predict", json={"input": []}).status_code == 422


def test_predict_refuse_des_vecteurs_de_longueurs_inegales(client, n_features):
    response = client.post(
        "/predict",
        json={"input": [[0.0] * n_features, [0.0] * (n_features - 1)]},
    )

    # Sans cette validation, numpy construit un tableau d'objets et l'API
    # renverrait une erreur 500 au lieu d'un refus explicite.
    assert response.status_code == 422


def test_predict_refuse_un_corps_sans_cle_input(client):
    assert client.post("/predict", json={"data": [[1.0]]}).status_code == 422


def test_route_inconnue_renvoie_un_404_json(client):
    """Un handler d'exception retournant un dict casserait la couche ASGI."""
    response = client.get("/cette-route-nexiste-pas")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "Endpoint non trouvé"
    assert body["path"] == "/cette-route-nexiste-pas"


# --------------------------------------------------------------------------- #
# /predict/form
# --------------------------------------------------------------------------- #
def test_predict_form_accepte_des_features_nommees(client):
    response = client.post("/predict/form", json={
        "mileage": 140000, "engine_power": 135,
        "model_key": "BMW", "fuel": "diesel", "car_type": "sedan",
        "has_gps": True,
    })

    assert response.status_code == 200
    assert len(response.json()["prediction"]) == 1


def test_predict_form_refuse_une_modalite_inconnue(client):
    response = client.post("/predict/form", json={"model_key": "Batmobile"})

    assert response.status_code == 400
    assert "Batmobile" in response.json()["detail"]


def test_predict_form_et_predict_donnent_le_meme_prix(client, api_module):
    """Les deux endpoints doivent partager exactement le même encodage."""
    features = api_module.feature_names
    vector = [0.0] * len(features)
    vector[features.index("mileage")] = 140000
    vector[features.index("engine_power")] = 135
    vector[features.index("model_key_BMW")] = 1.0

    positional = client.post("/predict", json={"input": [vector]}).json()
    named = client.post("/predict/form", json={
        "mileage": 140000, "engine_power": 135, "model_key": "BMW",
    }).json()

    assert positional == named


# --------------------------------------------------------------------------- #
# Documentation et métadonnées
# --------------------------------------------------------------------------- #
def test_docs_contient_un_titre_h1(client):
    """Exigence explicite de l'énoncé : « An h1 title »."""
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<h1>" in response.text


def test_docs_documente_chaque_endpoint(client):
    page = client.get("/docs").text

    for endpoint in ["/predict", "/health", "/model-info", "/features", "/version"]:
        assert endpoint in page, f"{endpoint} absent de la documentation"
    for keyword in ["POST", "GET", "Entrée requise", "Sortie attendue"]:
        assert keyword in page


def test_swagger_reste_accessible(client):
    assert client.get("/swagger").status_code == 200


def test_page_daccueil(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health(client, api_module):
    body = client.get("/health").json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["features_count"] == len(api_module.feature_names)


def test_model_info_expose_des_metriques_coherentes(client, api_module):
    body = client.get("/model-info").json()

    assert body["features_count"] == len(body["feature_names"])
    assert 0 < body["metrics"]["r2_test"] <= 1
    assert body["metrics"]["rmse_test"] > 0
    assert body["sklearn_version"] == api_module.model_package["sklearn_version"]


def test_features_liste_les_colonnes_attendues(client, n_features):
    body = client.get("/features").json()

    assert body["count"] == n_features
    assert len(body["features"]) == n_features
    # La colonne d'index du CSV ne doit jamais réapparaître comme feature.
    assert "Unnamed: 0" not in body["features"]
    assert "mileage" in body["features"]
    assert "engine_power" in body["features"]


def test_version(client):
    body = client.get("/version").json()

    assert body["api_version"]
    assert body["sklearn_version"]


# --------------------------------------------------------------------------- #
# Non-régression du modèle
# --------------------------------------------------------------------------- #
def test_le_modele_natteint_pas_un_r2_degrade(api_module):
    assert api_module.model_metrics["r2_test"] > 0.70


def test_la_puissance_moteur_augmente_le_prix(client, api_module):
    """Contrôle de bon sens métier sur le comportement du modèle."""
    features = api_module.feature_names
    base = [0.0] * len(features)
    base[features.index("mileage")] = 140000

    weak, strong = list(base), list(base)
    weak[features.index("engine_power")] = 75
    strong[features.index("engine_power")] = 250

    prices = client.post("/predict", json={"input": [weak, strong]}).json()["prediction"]
    assert prices[1] > prices[0]


@pytest.mark.parametrize("endpoint", ["/health", "/model-info", "/features", "/version"])
def test_les_endpoints_get_repondent(client, endpoint):
    assert client.get(endpoint).status_code == 200
