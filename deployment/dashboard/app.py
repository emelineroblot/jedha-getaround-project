"""
GetAround — Dashboard d'aide à la décision sur le délai minimum entre locations.

Répond aux quatre questions du Product Manager :
  1. Quelle part du revenu des propriétaires serait affectée ?
  2. Combien de locations seraient impactées, selon le seuil et le périmètre ?
  3. À quelle fréquence les conducteurs sont-ils en retard, et avec quel impact
     sur le conducteur suivant ?
  4. Combien de cas problématiques seraient résolus ?

Toute la logique de calcul vit dans `analysis.py`, partagé avec le notebook
d'exploration : ce fichier ne contient que la couche de présentation.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis

st.set_page_config(
    page_title="GetAround — Analyse des retards",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCOPE_LABELS = {
    "Tous les véhicules": "all",
    "Uniquement Connect": "connect",
    "Uniquement Mobile": "mobile",
}


def fr(number: float, decimals: int = 0) -> str:
    """Formate un nombre à la française : espace insécable fine comme séparateur."""
    return f"{number:,.{decimals}f}".replace(",", " ")



# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Chargement des données GetAround…")
def get_data() -> tuple[pd.DataFrame, float]:
    """Charge et enrichit les données ; retourne aussi le prix moyen journalier."""
    delays = analysis.enrich(analysis.load_delay_data())
    avg_price = float(analysis.load_pricing_data()["rental_price_per_day"].mean())
    return delays, avg_price


try:
    df, AVG_PRICE = get_data()
except Exception as exc:
    st.error(
        f"Impossible de charger les données ({type(exc).__name__}: {exc}).\n\n"
        f"Le dashboard télécharge automatiquement les fichiers depuis "
        f"`{analysis.S3_BASE}` au premier lancement : vérifiez la connexion réseau."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar : paramètres de décision
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Paramètres de simulation")

scope_label = st.sidebar.radio(
    "Périmètre d'application",
    options=list(SCOPE_LABELS),
    help="Sur quels véhicules la fonctionnalité serait-elle activée ?",
)
scope = SCOPE_LABELS[scope_label]

threshold = st.sidebar.slider(
    "Délai minimum entre deux locations (minutes)",
    min_value=0, max_value=720, value=120, step=30,
)
st.sidebar.caption(f"Soit **{threshold / 60:.1f} heure(s)**.")

st.sidebar.markdown("---")
st.sidebar.subheader("Hypothèses économiques")
st.sidebar.caption(
    "Le seuil « optimal » dépend de la valeur que l'on accorde à un incident "
    "évité. Ces deux curseurs rendent cet arbitrage explicite plutôt que caché "
    "dans une formule."
)
cost_per_incident = st.sidebar.slider(
    "Coût d'un incident (€)", min_value=0, max_value=600, value=150, step=25,
    help="Friction client, support, geste commercial, risque d'annulation.",
)
displacement_rate = st.sidebar.slider(
    "Taux de report des locations bloquées (%)", min_value=0, max_value=100, value=50, step=5,
    help=(
        "Part des réservations refusées qui se reporteraient sur un autre "
        "créneau ou une autre voiture : ce chiffre d'affaires n'est pas perdu."
    ),
) / 100

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Jeu de données** : {len(df):,} locations\n\n"
    f"**Prix moyen observé** : {AVG_PRICE:.2f} €/jour\n\n"
    "Source : datasets officiels GetAround (Jedha)."
        .replace(",", " ")
)

# Vues dérivées du périmètre choisi
scoped = analysis.apply_scope(df, scope)
late = analysis.late_stats(scoped)
problems = analysis.problematic_stats(scoped)
simulation = analysis.simulate_thresholds(df, scope=scope, avg_price_per_day=AVG_PRICE)
arbitrage = analysis.optimal_threshold(simulation, cost_per_incident, displacement_rate, AVG_PRICE)


# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
st.title("🚗 GetAround — Délai minimum entre deux locations")
st.markdown(
    "Ce tableau de bord évalue l'arbitrage entre **la friction subie par le "
    "conducteur suivant** et **le chiffre d'affaires sacrifié** lorsqu'on impose "
    "un délai minimum entre deux locations d'un même véhicule."
)

with st.expander("📐 Comment un « cas problématique » est-il défini ?"):
    st.markdown(
        f"""
Une location est problématique lorsque **le retard de la location précédente
dépasse le battement disponible** avant le checkin de la suivante :

```python
previous_delay = df.set_index('rental_id')['delay_at_checkout_in_minutes']
df['previous_delay_in_minutes'] = df['previous_ended_rental_id'].map(previous_delay)
df['is_problematic'] = (
    df['previous_delay_in_minutes'] > df['time_delta_with_previous_rental_in_minutes']
)
```

Le retard qui compte est donc celui **de la location précédente**, ramené par
jointure sur `previous_ended_rental_id` — et non le retard de la location
courante, qui ne dit rien de la gêne subie par son conducteur.

**Limites assumées** : sur les {problems['n_consecutive']:,} locations
consécutives, {problems['n_previous_delay_unknown']} n'ont pas de retard connu
pour la location précédente (checkout non renseigné ou location annulée) ; elles
ne peuvent pas être classées et sont comptées comme non problématiques.
        """.replace(",", " ")
    )


# --------------------------------------------------------------------------- #
# 1. Fréquence des retards
# --------------------------------------------------------------------------- #
st.header("1 · À quelle fréquence les conducteurs sont-ils en retard ?")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locations analysées", fr(late['n_total']))
col2.metric(
    "Taux de retard", f"{late['late_pct']:.1f} %",
    help=(
        f"Calculé sur les {late['n_known']:,} locations dont le checkout est "
        f"renseigné. Les {late['n_unknown']:,} autres "
        f"({late['unknown_pct']:.1f} %) sont indéterminées, et non « à l'heure »."
    ).replace(",", " "),
)
col3.metric(
    "Retard médian", f"{late['median_delay']:.0f} min",
    help="La médiane, pas la moyenne : la distribution est très asymétrique.",
)
col4.metric(
    "Retard moyen", f"{late['mean_delay']:.0f} min",
    help=(
        f"Tiré vers le haut par des valeurs extrêmes (maximum observé : "
        f"{late['max_delay']:,.0f} min). À ne pas utiliser en communication."
    ).replace(",", " "),
)

st.caption(
    f"⚠️ {late['n_unknown']:,} locations ({late['unknown_pct']:.1f} %) n'ont pas de "
    f"donnée de checkout — majoritairement des locations annulées. Les inclure au "
    f"dénominateur ferait mécaniquement chuter le taux de retard affiché."
    .replace(",", " ")
)

col_left, col_right = st.columns(2)

with col_left:
    # Bornage explicite : sans cela, les retards extrêmes (jusqu'à ~49 jours)
    # écrasent toute la distribution dans le premier bin.
    p99 = df["delay_at_checkout_in_minutes"].quantile(0.99)
    has_value = scoped["delay_at_checkout_in_minutes"].notna()
    visible = has_value & scoped["delay_at_checkout_in_minutes"].between(-720, p99)
    # Les valeurs manquantes ne sont pas « hors cadre » : elles n'existent pas.
    out_of_frame = int((has_value & ~visible).sum())
    fig_hist = px.histogram(
        scoped[visible], x="delay_at_checkout_in_minutes", nbins=80,
        title="Distribution des retards au checkout",
        labels={"delay_at_checkout_in_minutes": "Retard (minutes)"},
        color_discrete_sequence=["#4b3fbb"],
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black",
                       annotation_text="Heure prévue")
    fig_hist.update_layout(height=380, showlegend=False, bargap=0.02)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"Affichage borné à [-12 h ; {p99:.0f} min] (99ᵉ centile). "
        f"{fr(out_of_frame)} location(s) hors cadre (retards extrêmes)."
    )

with col_right:
    buckets = (
        scoped[scoped["has_checkout_info"]]["delay_bucket"]
        .value_counts()
        .reindex([
            "À l'heure ou en avance",
            "Retard court (< 30 min)",
            "Retard modéré (30 min - 2 h)",
            "Retard structurel (> 2 h)",
        ])
        .dropna()
    )
    fig_buckets = px.pie(
        values=buckets.values, names=buckets.index, hole=0.4,
        title="Nature des retards (checkout renseigné uniquement)",
        color_discrete_sequence=["#10b981", "#fbbf24", "#f97316", "#dc2626"],
    )
    fig_buckets.update_layout(height=380)
    st.plotly_chart(fig_buckets, use_container_width=True)
    st.caption(
        "Tous les retards ne se valent pas : un retard court est absorbable, "
        "un retard structurel ne l'est pas."
    )

st.subheader("Comparaison Mobile / Connect")
by_type = analysis.late_stats_by_checkin_type(df)
col_a, col_b = st.columns([3, 2])
with col_a:
    fig_type = px.bar(
        by_type, x="Type", y="% de retards", text="% de retards",
        color="% de retards", color_continuous_scale="Reds",
        title="Taux de retard par type de checkin",
    )
    fig_type.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig_type.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_type, use_container_width=True)
with col_b:
    st.dataframe(
        by_type.style.format({
            "Locations": "{:,.0f}", "Checkout renseigné": "{:,.0f}",
            "En retard": "{:,.0f}", "% de retards": "{:.1f} %",
            "Retard médian (min)": "{:.0f}", "Retard moyen (min)": "{:.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

connect_row = by_type[by_type["Type"] == "connect"]
mobile_row = by_type[by_type["Type"] == "mobile"]
if not connect_row.empty and not mobile_row.empty:
    st.info(
        f"**Connect est nettement plus fiable** : {connect_row['% de retards'].iloc[0]:.1f} % "
        f"de retards contre {mobile_row['% de retards'].iloc[0]:.1f} % en Mobile, avec un retard "
        f"médian de {connect_row['Retard médian (min)'].iloc[0]:.0f} min contre "
        f"{mobile_row['Retard médian (min)'].iloc[0]:.0f} min. "
        "C'est précisément ce qui rend contre-intuitif de déployer la mesure sur Connect "
        "en premier — voir la section 5."
    )


# --------------------------------------------------------------------------- #
# 2. Impact sur le conducteur suivant
# --------------------------------------------------------------------------- #
st.header("2 · Quel impact sur le conducteur suivant ?")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Locations consécutives", fr(problems['n_consecutive']),
    f"{problems['consecutive_pct']:.1f} % du parc",
    delta_color="off",
    help="Seules ces locations peuvent être affectées par la fonctionnalité.",
)
col2.metric(
    "Cas problématiques", fr(problems['n_problematic']),
    f"{problems['problematic_pct_consecutive']:.1f} % des consécutives",
    delta_color="inverse",
)
col3.metric(
    "Soit, sur tout le parc", f"{problems['problematic_pct_total']:.2f} %",
    help="Le même chiffre rapporté à l'ensemble des locations.",
)
col4.metric(
    "Attente médiane subie", f"{problems['median_wait']:.0f} min",
    help=f"Moyenne : {problems['mean_wait']:.0f} min ; 90ᵉ centile : {problems['p90_wait']:.0f} min.",
)

st.subheader("Les retards font-ils annuler la location suivante ?")
cancel = analysis.cancellation_impact(scoped)
col_a, col_b = st.columns([2, 3])
with col_a:
    st.dataframe(
        cancel.style.format({
            "Locations": "{:,.0f}", "Annulées": "{:,.0f}",
            "Taux d'annulation (%)": "{:.1f} %",
        }),
        use_container_width=True, hide_index=True,
    )
with col_b:
    fig_cancel = px.bar(
        cancel, x="Situation", y="Taux d'annulation (%)",
        text="Taux d'annulation (%)", color="Situation",
        color_discrete_sequence=["#10b981", "#dc2626"],
        title="Taux d'annulation selon l'exposition au retard du précédent",
    )
    fig_cancel.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig_cancel.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_cancel, use_container_width=True)

if len(cancel) == 2:
    delta = cancel["Taux d'annulation (%)"].iloc[1] - cancel["Taux d'annulation (%)"].iloc[0]
    st.success(
        f"Un conducteur qui subit le retard du précédent annule "
        f"**{delta:+.1f} points** plus souvent "
        f"({cancel['Taux d\'annulation (%)'].iloc[1]:.1f} % contre "
        f"{cancel['Taux d\'annulation (%)'].iloc[0]:.1f} %). "
        "C'est le lien chiffré entre retard et perte de chiffre d'affaires — "
        "corrélation observée, la causalité n'est pas établie sur ces seules données."
    )


# --------------------------------------------------------------------------- #
# 3. Simulation
# --------------------------------------------------------------------------- #
st.header("3 · Combien de locations seraient affectées, et combien de cas résolus ?")

current = simulation.set_index("Seuil (min)")
nearest = min(simulation["Seuil (min)"], key=lambda t: abs(t - threshold))
row = current.loc[nearest]
if nearest != threshold:
    st.caption(f"Seuil le plus proche simulé : **{nearest} min**.")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Locations bloquées", fr(row['Locations bloquées']),
    f"{row['% des consécutives']:.1f} % des consécutives",
    delta_color="inverse",
)
col2.metric(
    "Rapporté au parc entier", f"{row['% du parc']:.2f} %",
    help="Le dénominateur qui compte pour évaluer le coût réel de la mesure.",
)
col3.metric(
    "Problèmes résolus", fr(row['Problèmes résolus']),
    f"{row['% problèmes résolus']:.1f} % des cas",
)
col4.metric(
    "CA potentiellement perdu",
    f"{fr(row['CA perdu (€)'])} €",
    help=f"Locations bloquées × prix moyen ({AVG_PRICE:.2f} €/jour), avant report.",
)

fig_tradeoff = go.Figure()
fig_tradeoff.add_trace(go.Scatter(
    x=simulation["Seuil (h)"], y=simulation["% problèmes résolus"],
    mode="lines+markers", name="Problèmes résolus (% des cas)",
    line=dict(color="#10b981", width=3),
))
fig_tradeoff.add_trace(go.Scatter(
    x=simulation["Seuil (h)"], y=simulation["% des consécutives"],
    mode="lines+markers", name="Locations bloquées (% des consécutives)",
    line=dict(color="#dc2626", width=3),
))
fig_tradeoff.add_trace(go.Scatter(
    x=simulation["Seuil (h)"], y=simulation["% du parc"],
    mode="lines+markers", name="Locations bloquées (% du parc)",
    line=dict(color="#f97316", width=3, dash="dot"),
))
fig_tradeoff.add_vline(
    x=threshold / 60, line_dash="dash", line_color="#4b3fbb",
    annotation_text=f"Seuil retenu : {threshold / 60:.1f} h",
)
fig_tradeoff.update_layout(
    title="Arbitrage : bénéfice client contre coût pour les propriétaires",
    xaxis_title="Délai minimum imposé (heures)", yaxis_title="Pourcentage (%)",
    hovermode="x unified", height=460, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
)
st.plotly_chart(fig_tradeoff, use_container_width=True)

st.caption(
    "Les deux courbes rouges mesurent la même chose avec deux dénominateurs "
    "différents : rapportée au parc entier, la mesure est bien moins coûteuse "
    "qu'elle n'en a l'air lorsqu'on la rapporte aux seules locations consécutives."
)

with st.expander("📋 Tableau détaillé de la simulation"):
    st.dataframe(
        simulation.style.format({
            "Seuil (h)": "{:.1f} h", "Locations bloquées": "{:,.0f}",
            "% des consécutives": "{:.1f} %", "% du parc": "{:.2f} %",
            "Problèmes résolus": "{:,.0f}", "% problèmes résolus": "{:.1f} %",
            "CA perdu (€)": "{:,.0f} €",
        }),
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------------------------------- #
# 4. Revenu
# --------------------------------------------------------------------------- #
st.header("4 · Quelle part du revenu des propriétaires serait affectée ?")

revenue = analysis.revenue_impact(df, threshold, scope=scope, avg_price_per_day=AVG_PRICE)
net_revenue = revenue["revenue_lost_eur"] * (1 - displacement_rate)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Part du revenu affectée", f"{revenue['revenue_share_pct']:.2f} %",
    help="Locations bloquées rapportées à toutes les locations du périmètre.",
)
col2.metric(
    "CA brut concerné", f"{fr(revenue['revenue_lost_eur'])} €",
    help="Avant prise en compte du report sur d'autres créneaux.",
)
col3.metric(
    "CA net perdu (après report)", f"{fr(net_revenue)} €",
    f"report de {displacement_rate:.0%}", delta_color="off",
)

st.warning(
    f"""
**Hypothèses de ce chiffrage** — à énoncer avec le résultat :

- Les deux jeux de données n'ont **aucune clé commune** (la table de pricing ne
  contient pas `car_id`) : le prix est approximé par la moyenne observée,
  **{AVG_PRICE:.2f} €/jour** (médiane 119 €, quartiles 104–136 €).
- La durée de location est supposée d'**un jour**, faute d'information dans le
  jeu de données sur les retards.
- Le chiffre est un **majorant** : une partie des réservations bloquées se
  reporterait sur un autre créneau ou un autre véhicule. C'est ce que modélise
  le curseur « taux de report » ({displacement_rate:.0%} actuellement).
"""
)


# --------------------------------------------------------------------------- #
# 5. Recommandation
# --------------------------------------------------------------------------- #
st.header("5 · Recommandation")

positive = arbitrage[arbitrage["Bénéfice net (€)"] > 0]
best = positive.loc[positive["Bénéfice net (€)"].idxmax()] if not positive.empty else None

col_left, col_right = st.columns([3, 2])

with col_left:
    if best is None:
        st.error(
            "Avec ces hypothèses, **aucun seuil n'est rentable** : le chiffre "
            "d'affaires sacrifié dépasse toujours la friction évitée. Relevez le "
            "coût d'un incident ou le taux de report pour explorer d'autres scénarios."
        )
    else:
        st.success(
            f"""
### Seuil recommandé : {best['Seuil (min)']:.0f} minutes ({best['Seuil (h)']:.1f} h)
**Périmètre : {scope_label.lower()}**

- Résout **{best['Problèmes résolus']:.0f} cas** sur
  {problems['n_problematic']}, soit **{best['% problèmes résolus']:.1f} %** de la friction
- Bloque **{best['Locations bloquées']:.0f} locations**, soit
  **{best['% du parc']:.2f} % du parc**
- Bénéfice net estimé : **{best['Bénéfice net (€)']:,.0f} €**
            """.replace(",", " ")
        )
        st.caption(
            f"Optimum calculé avec un coût d'incident de {cost_per_incident} € et "
            f"un taux de report de {displacement_rate:.0%}. Modifiez ces hypothèses "
            "dans la barre latérale : la recommandation se recalcule."
        )

with col_right:
    fig_net = px.bar(
        arbitrage, x="Seuil (min)", y="Bénéfice net (€)",
        color=arbitrage["Bénéfice net (€)"] > 0,
        color_discrete_map={True: "#10b981", False: "#dc2626"},
        title="Bénéfice net par seuil",
    )
    fig_net.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_net, use_container_width=True)

st.subheader("Le périmètre : faut-il commencer par Connect ?")

comparison = []
for label, code in SCOPE_LABELS.items():
    sim = analysis.simulate_thresholds(df, scope=code, avg_price_per_day=AVG_PRICE)
    arb = analysis.optimal_threshold(sim, cost_per_incident, displacement_rate, AVG_PRICE)
    pos = arb[arb["Bénéfice net (€)"] > 0]
    top = pos.loc[pos["Bénéfice net (€)"].idxmax()] if not pos.empty else None
    stats = analysis.problematic_stats(analysis.apply_scope(df, code))
    comparison.append({
        "Périmètre": label,
        "Locations": stats["n_total"],
        "Consécutives": stats["n_consecutive"],
        "Cas problématiques": stats["n_problematic"],
        "% des consécutives": stats["problematic_pct_consecutive"],
        "Seuil optimal (min)": top["Seuil (min)"] if top is not None else 0,
        "Bénéfice net (€)": top["Bénéfice net (€)"] if top is not None else 0.0,
    })

st.dataframe(
    pd.DataFrame(comparison).style.format({
        "Locations": "{:,.0f}", "Consécutives": "{:,.0f}",
        "Cas problématiques": "{:,.0f}", "% des consécutives": "{:.1f} %",
        "Seuil optimal (min)": "{:.0f}", "Bénéfice net (€)": "{:,.0f} €",
    }),
    use_container_width=True, hide_index=True,
)

st.markdown(
    """
**Lecture.** Connect concentre moins de friction que Mobile : y déployer la
mesure en premier coûterait du chiffre d'affaires là où le problème est le moins
présent. Un **seuil différencié** — plus court sur Connect, plus long sur Mobile —
capte davantage de friction évitée à coût égal qu'un seuil uniforme.

**Mise en garde méthodologique.** Cette simulation mesure des locations
*observées* qui n'auraient pas eu lieu. Elle ne modélise pas le report de la
demande, ni le fait qu'un conducteur ayant une meilleure expérience réserve
davantage ensuite. Un test A/B reste nécessaire avant généralisation.
"""
)

with st.expander("🔍 Données brutes et statistiques descriptives"):
    st.dataframe(scoped.head(200), use_container_width=True)
    st.dataframe(
        scoped[[
            "delay_at_checkout_in_minutes",
            "time_delta_with_previous_rental_in_minutes",
            "previous_delay_in_minutes",
            "wait_for_next_driver",
        ]].describe(),
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    f"GetAround Analysis · {len(df):,} locations · prix moyen {AVG_PRICE:.2f} €/jour "
    f"· projet Jedha".replace(",", " ")
)
