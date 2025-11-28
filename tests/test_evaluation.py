"""Tests pour le module evaluation."""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.evaluation import (
    restore, output_to_mask, pixel_error, rand_error
)


def test_restore():
    """Test : Restauration des images."""
    print("Test 1: Image restoration...")
    
    # Créer une image normalisée
    images = torch.randn(2, 1, 100, 100)
    restored = restore(images)
    
    assert restored.shape[0] == 2, "Should have 2 images"
    assert restored.shape[1] == 100, "Height should be 100"
    assert restored.shape[2] == 100, "Width should be 100"
    
    print(f"✓ Restoration OK")
    print(f"  Input shape:  {images.shape}")
    print(f"  Output shape: {restored.shape}")


def test_output_to_mask():
    """Test : Conversion sortie -> masque."""
    print("\nTest 2: Output to mask conversion...")
    
    # Simuler une sortie du modèle (logits)
    outputs = torch.randn(2, 2, 100, 100)  # 2 classes
    masks = output_to_mask(outputs)
    
    assert masks.shape[0] == 2, "Should have 2 masks"
    assert len(masks.shape) == 3, "Should be 3D (B, H, W)"
    assert masks.dtype == torch.uint8, "Should be uint8"
    assert torch.all((masks == 0) | (masks == 1)), "Values should be 0 or 1"
    
    print(f"✓ Conversion OK")
    print(f"  Input shape:  {outputs.shape}")
    print(f"  Output shape: {masks.shape}")
    print(f"  Unique values: {masks.unique()}")


def test_pixel_error():
    """Test : Calcul de l'erreur pixel."""
    print("\nTest 3: Pixel error calculation...")
    
    # Masques identiques -> erreur = 0
    gt1 = torch.zeros(100, 100, dtype=torch.uint8)
    pred1 = torch.zeros(100, 100, dtype=torch.uint8)
    error1 = pixel_error(gt1, pred1)
    assert error1 == 0.0, "Error should be 0 for identical masks"
    
    # Masques complètement différents -> erreur = 1
    gt2 = torch.zeros(100, 100, dtype=torch.uint8)
    pred2 = torch.ones(100, 100, dtype=torch.uint8)
    error2 = pixel_error(gt2, pred2)
    assert error2 == 1.0, "Error should be 1 for completely different masks"
    
    # Masques partiellement différents
    gt3 = torch.zeros(100, 100, dtype=torch.uint8)
    pred3 = torch.zeros(100, 100, dtype=torch.uint8)
    pred3[:50, :] = 1  # 50% différent
    error3 = pixel_error(gt3, pred3)
    assert 0 < error3 < 1, "Error should be between 0 and 1"
    
    print(f"✓ Pixel error OK")
    print(f"  Identical masks error:   {error1:.4f}")
    print(f"  Opposite masks error:    {error2:.4f}")
    print(f"  Partial difference:      {error3:.4f}")


def test_rand_error():
    """Test : Calcul du Rand error."""
    print("\nTest 4: Rand error calculation...")
    
    # Masques identiques -> erreur proche de 0
    gt1 = torch.zeros(50, 50, dtype=torch.uint8)
    pred1 = torch.zeros(50, 50, dtype=torch.uint8)
    error1 = rand_error(gt1, pred1)
    assert error1 < 0.01, f"Error should be near 0, got {error1}"
    
    # Masques différents -> erreur > 0
    gt2 = torch.zeros(50, 50, dtype=torch.uint8)
    pred2 = torch.ones(50, 50, dtype=torch.uint8)
    error2 = rand_error(gt2, pred2)
    assert error2 > 0, "Error should be positive"
    
    print(f"✓ Rand error OK")
    print(f"  Identical masks error: {error1:.4f}")
    print(f"  Different masks error: {error2:.4f}")


def test_error_shape_validation():
    """Test : Validation des formes."""
    print("\nTest 5: Shape validation...")
    
    gt = torch.zeros(100, 100, dtype=torch.uint8)
    pred_wrong = torch.zeros(50, 50, dtype=torch.uint8)
    
    try:
        pixel_error(gt, pred_wrong)
        assert False, "Should raise ValueError for different shapes"
    except ValueError:
        print(f"✓ Shape validation OK (ValueError raised as expected)")


def run_all_tests():
    """Exécute tous les tests d'évaluation."""
    print("="*60)
    print("TESTS EVALUATION MODULE")
    print("="*60)
    
    try:
        test_restore()
        test_output_to_mask()
        test_pixel_error()
        test_rand_error()
        test_error_shape_validation()
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