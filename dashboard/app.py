"""
🚗 GetAround - Dashboard d'Analyse des Retards
Analyse des retards au checkout et simulation de seuils de délai minimum
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ===== CONFIGURATION PAGE =====
st.set_page_config(
    page_title="GetAround Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CHARGEMENT DES DONNÉES =====
@st.cache_data
def load_data():
    """Charger les données depuis le fichier Excel"""
    try:
        df = pd.read_excel('../data/get_around_delay_analysis.xlsx')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable. Assurez-vous que 'get_around_delay_analysis.xlsx' est dans le dossier 'data/'")
        st.stop()

# Charger les données
df = load_data()

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

# Filtres
st.sidebar.subheader("Filtres")
checkin_types = st.sidebar.multiselect(
    "Type de checkin",
    options=df['checkin_type'].unique(),
    default=df['checkin_type'].unique(),
    help="Filtrer par type de checkin (Mobile, Connect)"
)

# Filtrer les données selon la sélection
df_filtered = df[df['checkin_type'].isin(checkin_types)] if checkin_types else df

st.sidebar.markdown("---")
st.sidebar.info("""
📊 **À propos**
Ce dashboard analyse les retards au checkout et aide à déterminer le seuil optimal de délai minimum entre deux locations.

**Données:** 21,310 locations
**Source:** GetAround
""")

# ===== TITRE PRINCIPAL =====
st.title("🚗 GetAround - Analyse des Retards")
st.markdown("""
**Objectif :** Déterminer le seuil optimal de délai minimum entre deux locations pour :
- ✅ Réduire les frictions pour les clients suivants
- ⚠️ Minimiser l'impact sur les revenus des propriétaires
""")
st.markdown("---")

# ===== SECTION 1 : MÉTRIQUES CLÉS =====
st.header("📊 Vue d'ensemble")

# Calculs
total_rentals = len(df_filtered)
late_rentals = (df_filtered['delay_at_checkout_in_minutes'] > 0).sum()
late_pct = (late_rentals / total_rentals * 100) if total_rentals > 0 else 0
avg_delay = df_filtered[df_filtered['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes'].mean()

df_with_next = df_filtered[df_filtered['time_delta_with_previous_rental_in_minutes'].notna()]
consecutive = len(df_with_next)

# Cas problématiques
df_with_next_copy = df_with_next.copy()
df_with_next_copy['is_problematic'] = (
    (df_with_next_copy['delay_at_checkout_in_minutes'] > 0) &
    (df_with_next_copy['delay_at_checkout_in_minutes'] > df_with_next_copy['time_delta_with_previous_rental_in_minutes'])
)
total_problems = df_with_next_copy['is_problematic'].sum()
problem_pct = (total_problems / consecutive * 100) if consecutive > 0 else 0

# Affichage des métriques
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Locations",
        f"{total_rentals:,}",
        help="Nombre total de locations dans la période"
    )

with col2:
    st.metric(
        "Locations en retard",
        f"{late_rentals:,}",
        f"{late_pct:.1f}%",
        delta_color="inverse",
        help="Pourcentage de locations avec un retard au checkout"
    )

with col3:
    st.metric(
        "Retard moyen",
        f"{avg_delay:.0f} min" if not pd.isna(avg_delay) else "N/A",
        f"{avg_delay/60:.1f}h" if not pd.isna(avg_delay) else "",
        help="Retard moyen pour les locations en retard"
    )

with col4:
    st.metric(
        "Cas problématiques",
        f"{total_problems:,}",
        f"{problem_pct:.1f}%",
        delta_color="inverse",
        help="Locations où le retard a impacté le client suivant"
    )

st.markdown("---")

# ===== SECTION 2 : DISTRIBUTIONS =====
st.header("📈 Distribution des retards")

col1, col2 = st.columns(2)

with col1:
    # Histogramme des retards
    fig_hist = px.histogram(
        df_filtered[df_filtered['delay_at_checkout_in_minutes'].notna()],
        x='delay_at_checkout_in_minutes',
        nbins=100,
        title="Distribution des retards au checkout",
        labels={'delay_at_checkout_in_minutes': 'Retard (minutes)', 'count': 'Nombre de locations'},
        color_discrete_sequence=['indianred']
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="À l'heure")
    fig_hist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Pie chart
    on_time = total_rentals - late_rentals
    fig_pie = go.Figure(data=[go.Pie(
        labels=['À l\'heure', 'En retard'],
        values=[on_time, late_rentals],
        marker_colors=['lightgreen', 'indianred'],
        hole=0.3
    )])
    fig_pie.update_layout(
        title="Proportion retard vs à l'heure",
        height=400,
        annotations=[dict(text=f'{late_pct:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ===== SECTION 3 : RETARDS PAR TYPE =====
st.markdown("---")
st.header("📱 Analyse par type de checkin")

# Calculs par type
checkin_stats = []
for checkin_type in df_filtered['checkin_type'].dropna().unique():
    subset = df_filtered[df_filtered['checkin_type'] == checkin_type]
    late_count = (subset['delay_at_checkout_in_minutes'] > 0).sum()
    late_pct_type = (late_count / len(subset) * 100) if len(subset) > 0 else 0
    avg_delay_type = subset[subset['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes'].mean() if late_count > 0 else 0
    median_delay_type = subset[subset['delay_at_checkout_in_minutes'] > 0]['delay_at_checkout_in_minutes'].median() if late_count > 0 else 0

    checkin_stats.append({
        'Type': checkin_type,
        'Total': len(subset),
        'Retards': late_count,
        'Pct_retards': late_pct_type,
        'Retard_moyen': avg_delay_type,
        'Retard_median': median_delay_type
    })

df_checkin = pd.DataFrame(checkin_stats)

col1, col2 = st.columns(2)

with col1:
    # Barplot % retards
    fig_bar1 = px.bar(
        df_checkin,
        x='Type',
        y='Pct_retards',
        title="Pourcentage de retards par type",
        labels={'Pct_retards': '% de retards', 'Type': 'Type de checkin'},
        color='Pct_retards',
        color_continuous_scale='Reds',
        text='Pct_retards'
    )
    fig_bar1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar1.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_bar1, use_container_width=True)

with col2:
    # Barplot retard moyen
    fig_bar2 = px.bar(
        df_checkin,
        x='Type',
        y='Retard_moyen',
        title="Retard moyen par type (minutes)",
        labels={'Retard_moyen': 'Retard moyen (min)', 'Type': 'Type de checkin'},
        color='Retard_moyen',
        color_continuous_scale='Blues',
        text='Retard_moyen'
    )
    fig_bar2.update_traces(texttemplate='%{text:.0f} min', textposition='outside')
    fig_bar2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_bar2, use_container_width=True)

# Tableau récapitulatif
st.subheader("Tableau récapitulatif par type")
st.dataframe(
    df_checkin.style.format({
        'Total': '{:,.0f}',
        'Retards': '{:,.0f}',
        'Pct_retards': '{:.1f}%',
        'Retard_moyen': '{:.0f} min',
        'Retard_median': '{:.0f} min'
    }),
    use_container_width=True
)

# ===== SECTION 4 : SIMULATEUR DE SEUILS =====
st.markdown("---")
st.header("🎯 Simulateur de Seuil Minimum")

st.markdown("""
Utilisez le simulateur ci-dessous pour voir l'impact de différents seuils de délai minimum entre deux locations.
""")

col1, col2 = st.columns([1, 3])

with col1:
    # Slider pour le seuil
    threshold = st.slider(
        "Seuil minimum (minutes)",
        min_value=0,
        max_value=720,
        value=120,
        step=30,
        help="Délai minimum imposé entre deux locations"
    )

    st.write(f"**{threshold} minutes** = **{threshold/60:.1f} heures**")

    # Scope
    scope = st.radio(
        "Périmètre d'application",
        options=["Tous les véhicules", "Uniquement Connect", "Uniquement Mobile"],
        index=0,
        help="Choisir sur quels véhicules appliquer le seuil"
    )

with col2:
    # Calcul de l'impact
    df_analysis = df_with_next_copy.copy()

    # Appliquer le scope
    if scope == "Uniquement Connect":
        df_analysis = df_analysis[df_analysis['checkin_type'] == 'connect']
    elif scope == "Uniquement Mobile":
        df_analysis = df_analysis[df_analysis['checkin_type'] == 'mobile']

    # Locations bloquées
    blocked = df_analysis[df_analysis['time_delta_with_previous_rental_in_minutes'] < threshold]

    # Problèmes résolus
    total_problems_scope = df_analysis['is_problematic'].sum()
    problems_solved = df_analysis[
        (df_analysis['is_problematic']) &
        (df_analysis['time_delta_with_previous_rental_in_minutes'] < threshold)
    ]

    # Métriques
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        blocked_pct = (len(blocked) / len(df_analysis) * 100) if len(df_analysis) > 0 else 0
        st.metric(
            "Locations bloquées",
            f"{len(blocked):,}",
            f"{blocked_pct:.1f}%",
            delta_color="inverse",
            help="Nombre de locations qui seraient refusées avec ce seuil"
        )

    with col_b:
        solved_pct = (len(problems_solved) / total_problems_scope * 100) if total_problems_scope > 0 else 0
        st.metric(
            "Problèmes résolus",
            f"{len(problems_solved):,}",
            f"{solved_pct:.1f}%",
            help="Nombre de cas problématiques qui seraient évités"
        )

    with col_c:
        revenue_impact = blocked_pct
        st.metric(
            "Impact revenu estimé",
            f"-{revenue_impact:.1f}%",
            delta_color="inverse",
            help="Estimation de l'impact sur les revenus (approximatif)"
        )

# Graphique comparatif
st.subheader("Comparaison de différents seuils")

# Générer les résultats pour différents seuils
thresholds_to_test = [0, 30, 60, 120, 180, 240, 360, 480, 720]
results = []

for t in thresholds_to_test:
    blocked_t = df_analysis[df_analysis['time_delta_with_previous_rental_in_minutes'] < t]
    problems_t = df_analysis[
        (df_analysis['is_problematic']) &
        (df_analysis['time_delta_with_previous_rental_in_minutes'] < t)
    ]

    results.append({
        'Seuil (min)': t,
        'Seuil (h)': t/60,
        'Locations bloquées (%)': (len(blocked_t)/len(df_analysis)*100) if len(df_analysis) > 0 else 0,
        'Problèmes résolus (%)': (len(problems_t)/total_problems_scope*100) if total_problems_scope > 0 else 0,
        'Locations bloquées': len(blocked_t),
        'Problèmes résolus': len(problems_t)
    })

df_results = pd.DataFrame(results)

# Graphique Trade-off
fig_tradeoff = go.Figure()

fig_tradeoff.add_trace(go.Scatter(
    x=df_results['Seuil (h)'],
    y=df_results['Locations bloquées (%)'],
    mode='lines+markers',
    name='Locations bloquées (%)',
    line=dict(color='red', width=3),
    marker=dict(size=10),
    hovertemplate='<b>Seuil</b>: %{x:.1f}h<br><b>Bloquées</b>: %{y:.1f}%<extra></extra>'
))

fig_tradeoff.add_trace(go.Scatter(
    x=df_results['Seuil (h)'],
    y=df_results['Problèmes résolus (%)'],
    mode='lines+markers',
    name='Problèmes résolus (%)',
    line=dict(color='green', width=3),
    marker=dict(size=10),
    hovertemplate='<b>Seuil</b>: %{x:.1f}h<br><b>Résolus</b>: %{y:.1f}%<extra></extra>'
))

# Ligne verticale pour le seuil actuel
fig_tradeoff.add_vline(
    x=threshold/60,
    line_dash="dash",
    line_color="blue",
    annotation_text=f"Seuil actuel: {threshold/60:.1f}h"
)

fig_tradeoff.update_layout(
    title="🎯 Trade-off : Locations bloquées vs Problèmes résolus",
    xaxis_title="Seuil (heures)",
    yaxis_title="Pourcentage (%)",
    hovermode='x unified',
    height=500,
    legend=dict(x=0.7, y=0.5)
)

st.plotly_chart(fig_tradeoff, use_container_width=True)

# Tableau des résultats
st.subheader("Tableau détaillé")
st.dataframe(
    df_results.style.format({
        'Seuil (h)': '{:.1f}h',
        'Locations bloquées (%)': '{:.1f}%',
        'Problèmes résolus (%)': '{:.1f}%',
        'Locations bloquées': '{:,.0f}',
        'Problèmes résolus': '{:,.0f}'
    }).background_gradient(subset=['Locations bloquées (%)'], cmap='Reds')
      .background_gradient(subset=['Problèmes résolus (%)'], cmap='Greens'),
    use_container_width=True
)

# ===== SECTION 5 : RECOMMANDATIONS =====
st.markdown("---")
st.header("💡 Insights & Recommandations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Observations clés")
    st.markdown(f"""
    - **{late_pct:.1f}%** des locations sont en retard
    - Le retard moyen est de **{avg_delay:.0f} minutes** ({avg_delay/60:.1f}h)
    - **{total_problems:,}** cas problématiques identifiés ({problem_pct:.1f}% des locations consécutives)
    - Le type **{df_checkin.loc[df_checkin['Pct_retards'].idxmax(), 'Type'] if len(df_checkin) > 0 else 'N/A'}** a le plus de retards
    """)

    st.info("""
    💡 **Insight important:**
    Seulement 8.6% des locations ont une location suivante immédiate, ce qui signifie que la majorité des retards n'impactent pas directement d'autres clients.
    """)

with col2:
    st.subheader("🎯 Recommandations")

    # Calcul du seuil optimal
    df_results['ratio'] = df_results['Problèmes résolus (%)'] / (df_results['Locations bloquées (%)'] + 1)
    optimal_idx = df_results['ratio'].idxmax()
    optimal_threshold = df_results.loc[optimal_idx]

    st.success(f"""
    **Seuil recommandé : {optimal_threshold['Seuil (min)']:.0f} minutes ({optimal_threshold['Seuil (h)']:.1f}h)**

    📊 **Impact attendu :**
    - ✅ Résout **{optimal_threshold['Problèmes résolus (%)']:.1f}%** des problèmes
    - ⚠️ Bloque **{optimal_threshold['Locations bloquées (%)']:.1f}%** des locations
    """)

    st.warning("""
    **Périmètre suggéré :**
    1. Commencer avec les voitures **Connect** uniquement
    2. Étendre progressivement selon les résultats
    3. Mesurer l'impact sur 3 mois avant généralisation
    """)

# ===== SECTION 6 : DONNÉES BRUTES =====
st.markdown("---")
with st.expander("📋 Voir les données brutes"):
    st.subheader("Aperçu des données")
    st.dataframe(df_filtered.head(100), use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df_filtered.describe(), use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 GetAround Analysis Dashboard | Données: 21,310 locations | Projet Jedha</p>
</div>
""", unsafe_allow_html=True)
