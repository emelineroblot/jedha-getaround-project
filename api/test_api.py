"""
Script de test pour l'API GetAround
Teste tous les endpoints de l'API
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def print_header(title):
    """Affiche un header stylisé"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)

def test_endpoint(method, endpoint, data=None, description=""):
    """Teste un endpoint et affiche le résultat"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n📍 {method} {endpoint}")
    if description:
        print(f"   {description}")

    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Méthode {method} non supportée")
            return

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            print(f"   ✅ Succès")
            result = response.json()
            print(f"   Réponse: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"   ❌ Erreur")
            print(f"   Réponse: {response.text}")

    except requests.exceptions.ConnectionError:
        print("   ❌ Erreur de connexion. L'API est-elle lancée ?")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

def main():
    """Fonction principale de test"""
    print("="*80)
    print("🚗 TEST DE L'API GETAROUND")
    print("="*80)
    print(f"URL de base: {BASE_URL}")

    # Test 1: Page d'accueil
    print_header("Test 1 : Page d'accueil")
    test_endpoint("GET", "/", description="Récupère la page HTML d'accueil")

    # Test 2: Health check
    print_header("Test 2 : Health Check")
    test_endpoint("GET", "/health", description="Vérifie que l'API fonctionne")

    # Test 3: Informations du modèle
    print_header("Test 3 : Informations du modèle")
    test_endpoint("GET", "/model-info", description="Récupère les infos du modèle ML")

    # Test 4: Liste des features
    print_header("Test 4 : Liste des features")
    test_endpoint("GET", "/features", description="Liste des features attendues")

    # Test 5: Version
    print_header("Test 5 : Version de l'API")
    test_endpoint("GET", "/version", description="Récupère la version")

    # Test 6: Prédiction (exemple simple)
    print_header("Test 6 : Prédiction avec un exemple")
    # Créer un exemple de données (56 features)
    example_data = {
        "input": [
            # Exemple : [index, mileage, engine_power, ...autres features...]
            [3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1] + [0]*46
        ]
    }
    test_endpoint("POST", "/predict", data=example_data, description="Prédit le prix d'un véhicule")

    # Test 7: Prédiction avec plusieurs véhicules
    print_header("Test 7 : Prédiction multiple")
    multi_data = {
        "input": [
            [3203, 109839, 135, 1, 1, 0, 0, 1, 0, 1] + [0]*46,
            [1500, 50000, 200, 1, 1, 1, 1, 1, 1, 1] + [0]*46,
        ]
    }
    test_endpoint("POST", "/predict", data=multi_data, description="Prédit le prix de 2 véhicules")

    # Test 8: Erreur - mauvais nombre de features
    print_header("Test 8 : Test d'erreur - Mauvais nombre de features")
    bad_data = {
        "input": [
            [100, 200, 300]  # Seulement 3 features au lieu de 56
        ]
    }
    test_endpoint("POST", "/predict", data=bad_data, description="Devrait retourner une erreur 400")

    # Test 9: Erreur - input vide
    print_header("Test 9 : Test d'erreur - Input vide")
    empty_data = {
        "input": []
    }
    test_endpoint("POST", "/predict", data=empty_data, description="Devrait retourner une erreur 422")

    # Test 10: Route inexistante
    print_header("Test 10 : Test 404 - Route inexistante")
    test_endpoint("GET", "/route-inexistante", description="Devrait retourner une erreur 404")

    # Résumé
    print("\n" + "="*80)
    print("✅ TESTS TERMINÉS")
    print("="*80)
    print("\nPour voir la documentation interactive, ouvrez :")
    print(f"   🌐 {BASE_URL}/docs")
    print("\nPour voir la page d'accueil :")
    print(f"   🌐 {BASE_URL}/")

if __name__ == "__main__":
    main()
