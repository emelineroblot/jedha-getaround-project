"""Tests de la logique d'analyse des retards.

Ces tests verrouillent les deux erreurs de méthode qui faussaient l'analyse
initiale : la jointure avec la location précédente et le dénominateur du taux
de retard.
"""

import pandas as pd
import pytest

import analysis


# --------------------------------------------------------------------------- #
# Jointure avec la location précédente
# --------------------------------------------------------------------------- #
def test_le_retard_precedent_vient_bien_de_la_location_precedente():
    """`previous_delay_in_minutes` doit valoir le retard de l'autre ligne."""
    df = pd.DataFrame({
        "rental_id": [1, 2],
        "car_id": [10, 10],
        "checkin_type": ["mobile", "mobile"],
        "state": ["ended", "ended"],
        # La location 1 rend la voiture avec 90 min de retard ; la 2 est à l'heure.
        "delay_at_checkout_in_minutes": [90.0, 0.0],
        "previous_ended_rental_id": [None, 1.0],
        "time_delta_with_previous_rental_in_minutes": [None, 60.0],
    })

    out = analysis.enrich(df)
    row = out[out["rental_id"] == 2].iloc[0]

    assert row["previous_delay_in_minutes"] == 90.0
    # 90 min de retard pour 60 min de battement -> le conducteur suivant attend.
    assert bool(row["is_problematic"]) is True
    assert row["wait_for_next_driver"] == 30.0


def test_le_retard_propre_ne_rend_pas_problematique():
    """Régression : comparer une ligne à son PROPRE retard est l'erreur d'origine."""
    df = pd.DataFrame({
        "rental_id": [1, 2],
        "car_id": [10, 10],
        "checkin_type": ["mobile", "mobile"],
        "state": ["ended", "ended"],
        # La location précédente est à l'heure ; c'est la 2 qui rend tard.
        "delay_at_checkout_in_minutes": [0.0, 300.0],
        "previous_ended_rental_id": [None, 1.0],
        "time_delta_with_previous_rental_in_minutes": [None, 60.0],
    })

    out = analysis.enrich(df)
    row = out[out["rental_id"] == 2].iloc[0]

    # Son propre retard de 300 min ne gêne personne en amont.
    assert bool(row["is_problematic"]) is False
    assert row["wait_for_next_driver"] == 0.0


def test_un_retard_precedent_inconnu_nest_pas_problematique():
    df = pd.DataFrame({
        "rental_id": [1, 2],
        "car_id": [10, 10],
        "checkin_type": ["mobile", "mobile"],
        "state": ["canceled", "ended"],
        "delay_at_checkout_in_minutes": [None, 0.0],
        "previous_ended_rental_id": [None, 1.0],
        "time_delta_with_previous_rental_in_minutes": [None, 60.0],
    })

    out = analysis.enrich(df)
    row = out[out["rental_id"] == 2].iloc[0]

    assert pd.isna(row["previous_delay_in_minutes"])
    assert bool(row["is_problematic"]) is False


# --------------------------------------------------------------------------- #
# Dénominateur du taux de retard
# --------------------------------------------------------------------------- #
def test_les_checkouts_manquants_ne_comptent_pas_comme_a_lheure():
    df = pd.DataFrame({
        "rental_id": [1, 2, 3, 4],
        "car_id": [1, 2, 3, 4],
        "checkin_type": ["mobile"] * 4,
        "state": ["ended", "ended", "canceled", "canceled"],
        "delay_at_checkout_in_minutes": [30.0, 0.0, None, None],
        "previous_ended_rental_id": [None] * 4,
        "time_delta_with_previous_rental_in_minutes": [None] * 4,
    })

    stats = analysis.late_stats(analysis.enrich(df))

    assert stats["n_known"] == 2
    assert stats["n_unknown"] == 2
    # 1 retard sur 2 checkouts connus = 50 %, et non 1 sur 4 = 25 %.
    assert stats["late_pct"] == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_data():
    return analysis.enrich(analysis.load_delay_data())


def test_la_simulation_est_monotone(real_data):
    """Un seuil plus élevé bloque plus et résout plus : jamais l'inverse."""
    results = analysis.simulate_thresholds(real_data)

    assert results["Locations bloquées"].is_monotonic_increasing
    assert results["Problèmes résolus"].is_monotonic_increasing


def test_un_seuil_nul_na_aucun_effet(real_data):
    row = analysis.simulate_thresholds(real_data, thresholds=[0]).iloc[0]

    assert row["Locations bloquées"] == 0
    assert row["Problèmes résolus"] == 0


def test_les_deux_denominateurs_different(real_data):
    """Le % du parc doit rester bien inférieur au % des consécutives."""
    row = analysis.simulate_thresholds(real_data, thresholds=[120]).iloc[0]

    assert row["% du parc"] < row["% des consécutives"]
    assert row["% du parc"] < 10


def test_la_part_de_revenu_croit_avec_le_seuil(real_data):
    low = analysis.revenue_impact(real_data, 30)
    high = analysis.revenue_impact(real_data, 360)

    assert high["revenue_share_pct"] > low["revenue_share_pct"]
    assert low["revenue_share_pct"] < 5  # la mesure reste marginale à l'échelle du parc


@pytest.mark.parametrize("scope", ["all", "connect", "mobile"])
def test_tous_les_perimetres_sont_simulables(real_data, scope):
    results = analysis.simulate_thresholds(real_data, scope=scope)

    assert len(results) == len(analysis.DEFAULT_THRESHOLDS)
    assert (results["% problèmes résolus"] <= 100.0001).all()


def test_un_perimetre_inconnu_est_refuse(real_data):
    with pytest.raises(ValueError):
        analysis.apply_scope(real_data, "connectt")


def test_larbitrage_economique_reagit_aux_hypotheses(real_data):
    results = analysis.simulate_thresholds(real_data)

    frugal = analysis.optimal_threshold(results, cost_per_incident=10)
    generous = analysis.optimal_threshold(results, cost_per_incident=1000)

    # Plus un incident coûte cher, plus un seuil élevé devient défendable.
    assert generous["Bénéfice net (€)"].idxmax() >= frugal["Bénéfice net (€)"].idxmax()


# --------------------------------------------------------------------------- #
# Cohérence du jeu de données réel
# --------------------------------------------------------------------------- #
def test_les_chiffres_cles_du_dataset(real_data):
    """Verrouille les chiffres cités dans le README et la soutenance."""
    late = analysis.late_stats(real_data)
    problems = analysis.problematic_stats(real_data)

    assert late["n_total"] == 21310
    assert late["n_known"] == 16346
    assert late["late_pct"] == pytest.approx(57.5, abs=0.1)
    assert problems["n_consecutive"] == 1841
    assert problems["n_problematic"] == 218
    assert problems["problematic_pct_consecutive"] == pytest.approx(11.8, abs=0.1)


def test_le_dataset_pricing_na_pas_de_colonne_index():
    """`index_col=0` doit neutraliser `Unnamed: 0`."""
    pricing = analysis.load_pricing_data()

    assert "Unnamed: 0" not in pricing.columns
    assert "rental_price_per_day" in pricing.columns
