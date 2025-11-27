import os
import sys
import tempfile
import numpy as np
from PIL import Image


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BiologyDataset, calculate_class_weights


def create_dummy_dataset():
    """Crée un dataset temporaire pour les tests."""
    temp_dir = tempfile.mkdtemp()
    
    # Créer structure de dossiers
    categories = ['benign', 'malignant', 'normal']
    for cat in categories:
        os.makedirs(os.path.join(temp_dir, cat), exist_ok=True)
    
    # Créer quelques images et masques factices
    for i, cat in enumerate(categories):
        for j in range(2):  # 2 images par catégorie
            # Image
            img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            Image.fromarray(img).save(
                os.path.join(temp_dir, cat, f'image_{i}_{j}.png')
            )
            
            # Masque
            mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
            Image.fromarray(mask).save(
                os.path.join(temp_dir, cat, f'image_{i}_{j}_mask.png')
            )
    
    return temp_dir


def test_dataset_creation():
    """Test : Création du dataset."""
    print("Test 1: Dataset creation...")
    temp_dir = create_dummy_dataset()
    
    try:
        dataset = BiologyDataset(temp_dir)
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"✓ Dataset créé avec {len(dataset)} images")
        print(f"  Catégories : {set(dataset.category)}")
    finally:
        # Nettoyage
        import shutil
        shutil.rmtree(temp_dir)


def test_dataset_getitem():
    """Test : Récupération d'un élément."""
    print("\nTest 2: Dataset __getitem__...")
    temp_dir = create_dummy_dataset()
    
    try:
        dataset = BiologyDataset(temp_dir)
        img, mask, weights = dataset[0]
        
        assert img is not None, "Image should not be None"
        assert mask is not None, "Mask should not be None"
        assert weights is not None, "Weights should not be None"
        
        print(f"✓ Item récupéré avec succès")
        print(f"  Image shape: {img.shape}")
        print(f"  Mask shape: {mask.shape}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_class_weights():
    """Test : Calcul des poids de classes."""
    print("\nTest 3: Class weights calculation...")
    temp_dir = create_dummy_dataset()
    
    try:
        weights = calculate_class_weights(temp_dir)
        assert isinstance(weights, dict), "Weights should be a dictionary"
        assert len(weights) > 0, "Weights should not be empty"
        
        print(f"✓ Poids calculés : {weights}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Exécute tous les tests du dataset."""
    print("="*60)
    print("TESTS DATASET MODULE")
    print("="*60)
    
    try:
        test_dataset_creation()
        test_dataset_getitem()
        test_class_weights()
        print("\n" + "="*60)
        print("✓ TOUS LES TESTS RÉUSSIS !")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n✗ ÉCHEC : {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)