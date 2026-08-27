# Données

Les deux datasets ne sont **pas versionnés** (voir `.gitignore`). Ils sont téléchargés
automatiquement à la première exécution du dashboard, des notebooks ou du script d'entraînement :
aucune étape manuelle n'est nécessaire après un clone.

## Sources officielles

| Fichier | Usage | URL |
|---|---|---|
| `get_around_delay_analysis.xlsx` | Analyse des retards | <https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx> |
| `get_around_pricing_project.csv` | Machine Learning | <https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv> |

## Téléchargement manuel

```bash
python -c "import sys; sys.path.insert(0, 'deployment/dashboard'); import analysis; analysis.load_delay_data(); analysis.load_pricing_data()"
```

## Schéma — `get_around_delay_analysis.xlsx` (21 310 lignes)

| Colonne | Type | Remplissage | Description |
|---|---|---|---|
| `rental_id` | int | 100 % | Identifiant de la location |
| `car_id` | int | 100 % | Identifiant du véhicule |
| `checkin_type` | str | 100 % | `mobile` ou `connect` |
| `state` | str | 100 % | `ended` (18 045) ou `canceled` (3 265) |
| `delay_at_checkout_in_minutes` | float | 76,7 % | Retard au checkout. **Manquant pour les locations annulées** : ne jamais compter ces lignes comme « à l'heure ». |
| `previous_ended_rental_id` | float | 8,6 % | Location précédente sur le même véhicule. **Clé de jointure indispensable** pour mesurer la friction subie. |
| `time_delta_with_previous_rental_in_minutes` | float | 8,6 % | Battement entre le checkin de cette location et le checkout prévu de la précédente. |

## Schéma — `get_around_pricing_project.csv` (4 843 lignes)

Colonne d'index parasite `Unnamed: 0` à neutraliser avec `index_col=0` : c'est un numéro de ligne,
pas une caractéristique du véhicule.

| Colonne | Type | Description |
|---|---|---|
| `model_key` | str | Marque (28 modalités) |
| `mileage` | int | Kilométrage — 1 valeur négative aberrante |
| `engine_power` | int | Puissance en chevaux — 1 valeur nulle aberrante |
| `fuel`, `paint_color`, `car_type` | str | Carburant, couleur, carrosserie |
| `private_parking_available`, `has_gps`, `has_air_conditioning`, `automatic_car`, `has_getaround_connect`, `has_speed_regulator`, `winter_tires` | bool | Équipements |
| `rental_price_per_day` | int | **Variable cible** — de 10 à 422 €, médiane 119 € |

## À noter

Les deux jeux de données n'ont **aucune clé commune** : la table de pricing ne contient pas de
`car_id`. Toute estimation de revenu doit donc passer par un prix moyen, et l'hypothèse doit être
énoncée explicitement.
