"""
Logique d'analyse des retards GetAround.

Ce module est la **source de vérité unique** : il est importé à la fois par le
dashboard Streamlit (`app.py`) et par le notebook `notebooks/01_EDA_delays.ipynb`.
Il ne dépend que de pandas/numpy afin de rester utilisable hors Streamlit.

Points de méthode importants
----------------------------
1. Le retard subi par un conducteur n'est PAS `delay_at_checkout_in_minutes` de sa
   propre ligne : c'est celui de la location précédente, récupéré via la jointure
   sur `previous_ended_rental_id`.
2. Le taux de retard se calcule sur les locations dont le checkout est renseigné
   (16 346 lignes), pas sur les 21 310 lignes du fichier : 23 % des valeurs sont
   manquantes et ne doivent pas être comptées comme « à l'heure ».
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Sources de données officielles Jedha
# --------------------------------------------------------------------------- #
S3_BASE = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/"
DELAY_FILENAME = "get_around_delay_analysis.xlsx"
PRICING_FILENAME = "get_around_pricing_project.csv"
DELAY_URL = S3_BASE + DELAY_FILENAME
PRICING_URL = S3_BASE + PRICING_FILENAME

# Seuils simulés par défaut (en minutes)
DEFAULT_THRESHOLDS = [0, 30, 60, 90, 120, 180, 240, 300, 360, 480, 600, 720]

# Périmètres d'application possibles pour la fonctionnalité
SCOPES = ("all", "connect", "mobile")

# Un retard de moins de 30 min est considéré comme tolérable par le métier ;
# au-delà de 2 h il devient structurel. Utilisé pour la segmentation.
SHORT_DELAY_MAX = 30
STRUCTURAL_DELAY_MIN = 120


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Dossier de cache local des datasets.

    Cherche d'abord `<repo>/data`, sinon `./data` à côté de ce fichier (cas du
    déploiement Hugging Face où le module est seul dans /app).
    """
    if data_dir is not None:
        return Path(data_dir)

    here = Path(__file__).resolve().parent
    repo_data = here.parent.parent / "data"  # deployment/dashboard/ -> repo/data
    if repo_data.is_dir():
        return repo_data
    return here / "data"


def _download_if_missing(filename: str, url: str, data_dir: Path) -> Path:
    """Télécharge le fichier depuis S3 s'il n'est pas déjà en cache local."""
    data_dir.mkdir(parents=True, exist_ok=True)
    local = data_dir / filename
    if not local.exists():
        import urllib.request

        urllib.request.urlretrieve(url, local)
    return local


def load_delay_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Charge `get_around_delay_analysis.xlsx` (téléchargement automatique)."""
    path = _download_if_missing(DELAY_FILENAME, DELAY_URL, _resolve_data_dir(data_dir))
    return pd.read_excel(path)


def load_pricing_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Charge `get_around_pricing_project.csv`.

    `index_col=0` neutralise la colonne d'index parasite `Unnamed: 0`, qui n'est
    pas une caractéristique du véhicule et ne doit jamais servir de feature.
    """
    path = _download_if_missing(PRICING_FILENAME, PRICING_URL, _resolve_data_dir(data_dir))
    return pd.read_csv(path, index_col=0)


# --------------------------------------------------------------------------- #
# Enrichissement
# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes dérivées nécessaires à toute l'analyse.

    Colonnes ajoutées
    -----------------
    has_checkout_info : bool
        Le checkout a-t-il été renseigné ? (23 % de valeurs manquantes)
    is_late : bool
        Cette location a-t-elle été rendue en retard ? (NaN -> False, mais
        toujours filtrer sur `has_checkout_info` avant de calculer un taux)
    delay_bucket : str
        Segmentation métier du retard.
    has_previous : bool
        Cette location suit-elle une autre location sur la même voiture ?
    previous_delay_in_minutes : float
        Retard de la location PRÉCÉDENTE, ramené ici par jointure sur
        `previous_ended_rental_id`. C'est LA colonne qui mesure la friction subie.
    is_problematic : bool
        Le retard du précédent dépasse le battement disponible -> le conducteur
        suivant a réellement dû attendre.
    wait_for_next_driver : float
        Minutes d'attente effectivement subies (0 si pas d'impact).
    """
    out = df.copy()

    # Sur un sous-ensemble où une de ces colonnes est entièrement vide, pandas
    # lui donne le dtype `object` et les comparaisons lèvent un TypeError.
    # On force donc le type numérique avant tout calcul.
    for column in (
        "delay_at_checkout_in_minutes",
        "previous_ended_rental_id",
        "time_delta_with_previous_rental_in_minutes",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["has_checkout_info"] = out["delay_at_checkout_in_minutes"].notna()
    out["is_late"] = out["delay_at_checkout_in_minutes"] > 0
    out["delay_bucket"] = _delay_bucket(out["delay_at_checkout_in_minutes"])

    out["has_previous"] = out["previous_ended_rental_id"].notna()

    # /!\ Cœur de l'analyse : on rapatrie le retard de la location PRÉCÉDENTE.
    previous_delay = out.set_index("rental_id")["delay_at_checkout_in_minutes"]
    out["previous_delay_in_minutes"] = out["previous_ended_rental_id"].map(previous_delay)

    out["is_problematic"] = (
        out["previous_delay_in_minutes"] > out["time_delta_with_previous_rental_in_minutes"]
    )
    out["wait_for_next_driver"] = (
        out["previous_delay_in_minutes"] - out["time_delta_with_previous_rental_in_minutes"]
    ).clip(lower=0)

    return out


def _delay_bucket(delay: pd.Series) -> pd.Series:
    """Segmente les retards en catégories lisibles par le métier."""
    conditions = [
        delay.isna(),
        delay <= 0,
        delay < SHORT_DELAY_MAX,
        delay < STRUCTURAL_DELAY_MIN,
    ]
    labels = ["Non renseigné", "À l'heure ou en avance", "Retard court (< 30 min)",
              "Retard modéré (30 min - 2 h)"]
    return pd.Series(
        np.select(conditions, labels, default="Retard structurel (> 2 h)"),
        index=delay.index,
    )


def apply_scope(df: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    """Restreint le périmètre d'application de la fonctionnalité.

    Le filtre porte sur le `checkin_type` de la location concernée. Dans 99,1 %
    des couples consécutifs, la location précédente a le même type (même voiture),
    le choix du côté du filtre est donc sans effet notable.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope doit être l'un de {SCOPES}, reçu {scope!r}")
    if scope == "all":
        return df
    return df[df["checkin_type"] == scope]


# --------------------------------------------------------------------------- #
# Statistiques descriptives
# --------------------------------------------------------------------------- #
def late_stats(df: pd.DataFrame) -> dict:
    """Statistiques de retard, calculées sur la bonne base.

    Le dénominateur est le nombre de locations dont le checkout est renseigné.
    Les valeurs manquantes sont comptées à part, jamais comme « à l'heure ».
    """
    known = df[df["has_checkout_info"]]
    late = known[known["is_late"]]

    n_total = len(df)
    n_known = len(known)
    n_late = len(late)

    return {
        "n_total": n_total,
        "n_known": n_known,
        "n_unknown": n_total - n_known,
        "unknown_pct": (n_total - n_known) / n_total * 100 if n_total else 0.0,
        "n_late": n_late,
        # /!\ dénominateur = locations avec checkout renseigné, pas n_total
        "late_pct": n_late / n_known * 100 if n_known else 0.0,
        "mean_delay": late["delay_at_checkout_in_minutes"].mean(),
        "median_delay": late["delay_at_checkout_in_minutes"].median(),
        "p90_delay": late["delay_at_checkout_in_minutes"].quantile(0.90),
        "max_delay": late["delay_at_checkout_in_minutes"].max(),
    }


def late_stats_by_checkin_type(df: pd.DataFrame) -> pd.DataFrame:
    """Même chose, ventilé par type de checkin."""
    rows = []
    for checkin_type in sorted(df["checkin_type"].dropna().unique()):
        stats = late_stats(df[df["checkin_type"] == checkin_type])
        rows.append({
            "Type": checkin_type,
            "Locations": stats["n_total"],
            "Checkout renseigné": stats["n_known"],
            "En retard": stats["n_late"],
            "% de retards": stats["late_pct"],
            "Retard médian (min)": stats["median_delay"],
            "Retard moyen (min)": stats["mean_delay"],
        })
    return pd.DataFrame(rows)


def problematic_stats(df: pd.DataFrame) -> dict:
    """Mesure la friction réellement subie par les conducteurs suivants."""
    consecutive = df[df["has_previous"]]
    known_prev = consecutive[consecutive["previous_delay_in_minutes"].notna()]
    impacted = consecutive[consecutive["is_problematic"]]

    return {
        "n_total": len(df),
        "n_consecutive": len(consecutive),
        "consecutive_pct": len(consecutive) / len(df) * 100 if len(df) else 0.0,
        "n_previous_delay_known": len(known_prev),
        "n_previous_delay_unknown": len(consecutive) - len(known_prev),
        "n_problematic": len(impacted),
        "problematic_pct_consecutive": len(impacted) / len(consecutive) * 100 if len(consecutive) else 0.0,
        "problematic_pct_total": len(impacted) / len(df) * 100 if len(df) else 0.0,
        "median_wait": impacted["wait_for_next_driver"].median(),
        "mean_wait": impacted["wait_for_next_driver"].mean(),
        "p90_wait": impacted["wait_for_next_driver"].quantile(0.90),
    }


def cancellation_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Taux d'annulation selon que le conducteur a subi un retard ou non.

    Répond au point de l'énoncé : « users that even had to cancel their rental
    because the car wasn't returned on time ».
    """
    consecutive = df[df["has_previous"]]
    table = (
        pd.crosstab(consecutive["is_problematic"], consecutive["state"], normalize="index") * 100
    )
    counts = pd.crosstab(consecutive["is_problematic"], consecutive["state"])

    out = pd.DataFrame({
        "Situation": ["Sans retard du précédent", "Retard du précédent subi"],
        "Locations": [counts.loc[False].sum(), counts.loc[True].sum()],
        "Annulées": [counts.loc[False].get("canceled", 0), counts.loc[True].get("canceled", 0)],
        "Taux d'annulation (%)": [
            table.loc[False].get("canceled", 0.0),
            table.loc[True].get("canceled", 0.0),
        ],
    })
    return out


# --------------------------------------------------------------------------- #
# Simulation de seuils
# --------------------------------------------------------------------------- #
def simulate_thresholds(
    df: pd.DataFrame,
    thresholds: list[int] | None = None,
    scope: str = "all",
    avg_price_per_day: float = 121.21,
) -> pd.DataFrame:
    """Simule l'effet d'un délai minimum entre deux locations.

    Une location est **bloquée** si le battement avec la location précédente est
    inférieur au seuil : elle n'aurait pas pu être réservée.
    Un cas problématique est **résolu** si la location concernée est bloquée.

    Les pourcentages sont fournis avec DEUX dénominateurs, car ils ne racontent
    pas la même histoire :
      - `% des consécutives` : intensité de la mesure là où elle s'applique ;
      - `% du parc`          : coût réel à l'échelle de la plateforme.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    scoped = apply_scope(df, scope)
    consecutive = scoped[scoped["has_previous"]]
    n_problematic = int(consecutive["is_problematic"].sum())
    n_scoped = len(scoped)
    n_consecutive = len(consecutive)

    rows = []
    for threshold in thresholds:
        blocked_mask = consecutive["time_delta_with_previous_rental_in_minutes"] < threshold
        solved_mask = consecutive["is_problematic"] & blocked_mask
        n_blocked = int(blocked_mask.sum())
        n_solved = int(solved_mask.sum())

        rows.append({
            "Seuil (min)": threshold,
            "Seuil (h)": threshold / 60,
            "Locations bloquées": n_blocked,
            "% des consécutives": n_blocked / n_consecutive * 100 if n_consecutive else 0.0,
            "% du parc": n_blocked / n_scoped * 100 if n_scoped else 0.0,
            "Problèmes résolus": n_solved,
            "% problèmes résolus": n_solved / n_problematic * 100 if n_problematic else 0.0,
            "CA perdu (€)": n_blocked * avg_price_per_day,
        })

    return pd.DataFrame(rows)


def revenue_impact(
    df: pd.DataFrame,
    threshold: int,
    scope: str = "all",
    avg_price_per_day: float = 121.21,
) -> dict:
    """Part du revenu propriétaire potentiellement affectée par la fonctionnalité.

    Question 1 de l'énoncé. Les deux datasets n'ayant aucune clé commune
    (la table pricing ne contient pas `car_id`), le prix est approximé par la
    moyenne du dataset pricing.

    Hypothèses — à énoncer explicitement :
      * durée de location supposée d'un jour (absente du dataset délais) ;
      * prix uniforme faute de jointure possible ;
      * majorant : une part des réservations bloquées se reporterait en pratique
        sur un autre créneau ou une autre voiture (cf. `displacement_rate`).
    """
    scoped = apply_scope(df, scope)
    consecutive = scoped[scoped["has_previous"]]
    n_blocked = int((consecutive["time_delta_with_previous_rental_in_minutes"] < threshold).sum())

    return {
        "n_blocked": n_blocked,
        "n_scope": len(scoped),
        "revenue_share_pct": n_blocked / len(scoped) * 100 if len(scoped) else 0.0,
        "revenue_lost_eur": n_blocked * avg_price_per_day,
        "avg_price_per_day": avg_price_per_day,
    }


def optimal_threshold(
    results: pd.DataFrame,
    cost_per_incident: float = 150.0,
    displacement_rate: float = 0.5,
    avg_price_per_day: float = 121.21,
) -> pd.DataFrame:
    """Arbitrage économique explicite entre friction évitée et CA sacrifié.

    On remplace le ratio arbitraire « résolus / bloqués » par un bénéfice net
    exprimé en euros :

        Gain = problèmes résolus x coût moyen d'un incident
        Coût = locations bloquées x prix moyen x (1 - taux de report)
        Net  = Gain - Coût

    `cost_per_incident` et `displacement_rate` sont des paramètres métier assumés,
    pas des constantes cachées : c'est au Product Manager de les fixer.
    """
    out = results.copy()
    out["Gain (€)"] = out["Problèmes résolus"] * cost_per_incident
    out["Coût (€)"] = out["Locations bloquées"] * avg_price_per_day * (1 - displacement_rate)
    out["Bénéfice net (€)"] = out["Gain (€)"] - out["Coût (€)"]
    return out
