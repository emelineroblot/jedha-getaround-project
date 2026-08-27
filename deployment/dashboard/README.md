---
title: GetAround Delay Analysis
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GetAround — Dashboard d'analyse des retards

Tableau de bord d'aide à la décision pour le Product Manager : faut-il imposer un délai minimum
entre deux locations, de quelle durée, et sur quel périmètre ?

> Ce dossier est déployé tel quel sur Hugging Face Spaces. Le code source de référence vit dans
> le dépôt GitHub du projet : <https://github.com/emelineroblot/jedha-getaround-project>

## Ce que le dashboard permet de faire

- Mesurer la **fréquence réelle des retards**, sur le bon dénominateur (les checkouts renseignés).
- Quantifier la **friction subie par le conducteur suivant**, via la jointure avec la location
  précédente.
- Simuler l'effet d'un **seuil** de 0 à 12 h, sur trois **périmètres** (tous / Connect / Mobile).
- Chiffrer la **part de revenu affectée**, en euros.
- Faire varier les deux **hypothèses économiques** — coût d'un incident, taux de report de la
  demande — et voir le seuil optimal se déplacer.

## Architecture

| Fichier | Rôle |
|---|---|
| `analysis.py` | Toute la logique de calcul. Importé à l'identique par le notebook d'exploration : une seule source de vérité, donc aucun écart possible entre l'analyse et le dashboard. |
| `app.py` | Couche de présentation Streamlit uniquement. |

## Données

Les deux datasets officiels sont téléchargés automatiquement depuis S3 au premier lancement, et
pré-téléchargés au moment du build Docker pour que le conteneur démarre sans dépendance réseau.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
