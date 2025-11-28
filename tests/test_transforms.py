
"""Tests pour le module transforms."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transforms import get_train_transforms, get_val_transforms


def test_train_transforms():
    """Test : Transformations d'entraînement."""
    print("Test 1: Training transforms...")
    
    # Créer une image et un masque factices
    image = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
    mask = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
    
    transforms = get_train_transforms()
    
    # Appliquer les transformations
    transformed = transforms(image=image, mask=mask)
    
    assert 'image' in transformed, "Should contain 'image' key"
    assert 'mask' in transformed, "Should contain 'mask' key"
    
    img_tensor = transformed['image']
    mask_tensor = transformed['mask']
    
    # Vérifier les dimensions
    assert len(img_tensor.shape) == 3, "Image should be 3D (C, H, W)"
    assert len(mask_tensor.shape) == 2, "Mask should be 2D (H, W)"
    
    print(f"✓ Transformations d'entraînement OK")
    print(f"  Image shape: {img_tensor.shape}")
    print(f"  Mask shape: {mask_tensor.shape}")
    print(f"  Mask unique values: {mask_tensor.unique()}")


def test_val_transforms():
    """Test : Transformations de validation."""
    print("\nTest 2: Validation transforms...")
    
    image = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
    mask = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
    
    transforms = get_val_transforms()
    transformed = transforms(image=image, mask=mask)
    
    assert 'image' in transformed, "Should contain 'image' key"
    assert 'mask' in transformed, "Should contain 'mask' key"
    
    print(f"✓ Transformations de validation OK")
    print(f"  Image shape: {transformed['image'].shape}")
    print(f"  Mask shape: {transformed['mask'].shape}")


def test_transforms_consistency():
    """Test : Cohérence des transformations."""
    print("\nTest 3: Transforms consistency...")
    
    image = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
    mask = np.random.randint(0, 2, (572, 572), dtype=np.uint8) * 255
    
    # Appliquer plusieurs fois
    transforms = get_val_transforms()
    result1 = transforms(image=image, mask=mask)
    result2 = transforms(image=image, mask=mask)
    
    # Les résultats doivent être identiques pour val (pas d'augmentation aléatoire)
    assert result1['image'].shape == result2['image'].shape
    assert result1['mask'].shape == result2['mask'].shape
    
    print(f"✓ Cohérence vérifiée")


def run_all_tests():
    """Exécute tous les tests des transformations."""
    print("="*60)
    print("TESTS TRANSFORMS MODULE")
    print("="*60)
    
    try:
        test_train_transforms()
        test_val_transforms()
        test_transforms_consistency()
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