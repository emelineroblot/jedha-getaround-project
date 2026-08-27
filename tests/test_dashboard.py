"""Test de fumée du dashboard Streamlit.

Vérifie que l'application s'exécute de bout en bout sans exception, sur le vrai
jeu de données et pour chaque périmètre. C'est ce test qui aurait détecté
l'absence de `matplotlib` ou un chemin de données invalide.
"""

from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "deployment" / "dashboard" / "app.py"
TIMEOUT = 180


@pytest.fixture(scope="module")
def app():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    return AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)


def test_le_dashboard_se_charge_sans_exception(app):
    result = app.run()

    assert not result.exception, [str(e) for e in result.exception]


def test_le_dashboard_affiche_ses_sections(app):
    result = app.run()

    headers = " ".join(h.value for h in result.header)
    for expected in ["retard", "conducteur suivant", "revenu", "Recommandation"]:
        assert expected.lower() in headers.lower(), f"section manquante : {expected}"


def test_le_taux_de_retard_affiche_est_le_bon(app):
    """Le KPI doit valoir 57,5 % — pas 44,1 %, qui compte les checkouts manquants."""
    result = app.run()

    values = [m.value for m in result.metric]
    assert any(value.startswith("57") for value in values), values


@pytest.mark.parametrize("scope_index", [0, 1, 2])
def test_chaque_perimetre_sexecute(app, scope_index):
    result = app.run()
    result.radio[0].set_value(result.radio[0].options[scope_index]).run()

    assert not result.exception, [str(e) for e in result.exception]


def test_le_curseur_de_seuil_ne_casse_rien(app):
    result = app.run()
    result.slider[0].set_value(360).run()

    assert not result.exception, [str(e) for e in result.exception]


def test_les_hypotheses_economiques_modifient_la_recommandation(app):
    """Faire varier le coût d'un incident doit changer le seuil recommandé."""
    result = app.run()

    result.slider[1].set_value(0).run()          # incident sans valeur
    cheap = " ".join(str(element.value) for element in result.success)

    result.slider[1].set_value(600).run()        # incident très coûteux
    expensive = " ".join(str(element.value) for element in result.success)

    assert cheap != expensive
