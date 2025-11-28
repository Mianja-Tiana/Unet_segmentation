"""Tests d'intégration complets."""
import os
import sys
import torch
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BiologyDataset
from models.unet import UNet, UNetLoss
from utils.transforms import get_train_transforms, get_val_transforms
from utils.evaluation import evaluate_error


def create_dummy_dataset():
    """Crée un dataset temporaire."""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'test'), exist_ok=True)
    
    for i in range(3):
        img = np.random.randint(0, 255, (572, 572), dtype=np.uint8)
        mask = np.random.randint(0, 2, (572, 572), dtype=np.uint8) * 255
        
        Image.fromarray(img).save(
            os.path.join(temp_dir, 'test', f'img_{i}.png')
        )
        Image.fromarray(mask).save(
            os.path.join(temp_dir, 'test', f'img_{i}_mask.png')
        )
    
    return temp_dir


def test_end_to_end_pipeline():
    """Test : Pipeline complet."""
    print("Test 1: End-to-end pipeline...")
    
    temp_dir = create_dummy_dataset()
    
    try:
        # 1. Créer dataset
        dataset = BiologyDataset(
            temp_dir,
            transforms=get_val_transforms()
        )
        assert len(dataset) > 0
        print(f"  ✓ Dataset créé ({len(dataset)} images)")
        
        # 2. Charger un batch
        img, mask, weights = dataset[0]
        assert img is not None
        print(f"  ✓ Batch chargé")
        
        # 3. Créer modèle
        model = UNet(n_channels=1, n_classes=2)
        print(f"  ✓ Modèle créé")
        
        # 4. Forward pass
        with torch.no_grad():
            output = model(img.unsqueeze(0))
        assert output.shape[1] == 2
        print(f"  ✓ Forward pass réussi")
        
        # 5. Calculer loss
        criterion = UNetLoss()
        loss = criterion(output, mask.unsqueeze(0), weights.unsqueeze(0))
        assert not torch.isnan(loss)
        print(f"  ✓ Loss calculée: {loss.item():.4f}")
        
        print(f"\n✓ Pipeline complet OK")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_training_step():
    """Test : Une étape d'entraînement."""
    print("\nTest 2: Training step simulation...")
    
    model = UNet(n_channels=1, n_classes=2)
    criterion = UNetLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Données factices
    images = torch.randn(2, 1, 572, 572)
    masks = torch.randint(0, 2, (2, 388, 388))
    weights = torch.ones(2, 388, 388)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks, weights)
    loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0
    print(f"✓ Training step OK (loss: {loss.item():.4f})")


def test_inference():
    """Test : Inférence."""
    print("\nTest 3: Inference simulation...")
    
    model = UNet(n_channels=1, n_classes=2)
    model.eval()
    
    with torch.no_grad():
        images = torch.randn(1, 1, 572, 572)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    assert predictions.shape[0] == 1
    assert len(predictions.shape) == 3
    print(f"✓ Inference OK")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Predictions shape: {predictions.shape}")


def test_batch_processing():
    """Test : Traitement par batch."""
    print("\nTest 4: Batch processing...")
    
    model = UNet(n_channels=1, n_classes=2)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    
    for bs in batch_sizes:
        with torch.no_grad():
            images = torch.randn(bs, 1, 572, 572)
            outputs = model(images)
        
        assert outputs.shape[0] == bs
        print(f"  ✓ Batch size {bs} OK")
    
    print(f"✓ Batch processing OK")


def test_model_save_load():
    """Test : Sauvegarde et chargement du modèle."""
    print("\nTest 5: Model save/load...")
    
    model1 = UNet(n_channels=1, n_classes=2)
    
    # Sauvegarder
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    torch.save(model1.state_dict(), temp_file.name)
    print(f"  ✓ Modèle sauvegardé")
    
    # Charger
    model2 = UNet(n_channels=1, n_classes=2)
    model2.load_state_dict(torch.load(temp_file.name))
    print(f"  ✓ Modèle chargé")
    
    # Vérifier que les poids sont identiques
    with torch.no_grad():
        x = torch.randn(1, 1, 572, 572)
        out1 = model1(x)
        out2 = model2(x)
        assert torch.allclose(out1, out2)
    
    os.unlink(temp_file.name)
    print(f"✓ Save/Load OK")


def run_all_tests():
    """Exécute tous les tests d'intégration."""
    print("="*60)
    print("TESTS D'INTÉGRATION")
    print("="*60)
    
    try:
        test_end_to_end_pipeline()
        test_training_step()
        test_inference()
        test_batch_processing()
        test_model_save_load()
        print("\n" + "="*60)
        print("✓ TOUS LES TESTS D'INTÉGRATION RÉUSSIS !")
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