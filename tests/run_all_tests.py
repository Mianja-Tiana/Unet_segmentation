"""Exécute tous les tests du projet."""
import sys
import subprocess


def run_test_file(test_file):
    """Exécute un fichier de test."""
    print(f"\n{'='*70}")
    print(f"Exécution : {test_file}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, f"tests/{test_file}"],
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Exécute tous les tests."""
    print("\n" + "="*70)
    print("EXÉCUTION DE TOUS LES TESTS")
    print("="*70)
    
    test_files = [
        "test_dataset.py",
        "test_transforms.py",
        "test_model.py",
        "test_evaluation.py",
        "test_integration.py"
    ]

    results = {}

    for test_file in test_files:
        success = run_test_file(test_file)
        results[test_file] = success

   
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS")
    print("="*70)

    total = len(results)
    passed = sum(results.values())

    for test_file, success in results.items():
        status = "✓ RÉUSSI" if success else "✗ ÉCHEC"
        print(f"{test_file:30s} : {status}")

    print(f"\n{passed}/{total} tests réussis")

    if passed == total:
        print("\n TOUS LES TESTS SONT PASSÉS ! ")
        return 0
    else:
        print(f"\n  {total - passed} test(s) ont échoué")
        return 1


if __name__ == "__main__":
    sys.exit(main())