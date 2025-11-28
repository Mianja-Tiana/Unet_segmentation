"""Tests pour le module model."""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet, UNetLoss, DoubleConv, Down, Up


def test_double_conv():
    """Test : Module DoubleConv."""
    print("Test 1: DoubleConv module...")
    
    module = DoubleConv(1, 64)
    x = torch.randn(1, 1, 572, 572)
    output = module(x)
    
    # Vérifier que les dimensions sont correctes (padding=0 réduit la taille)
    assert output.shape[1] == 64, f"Expected 64 channels, got {output.shape[1]}"
    assert output.shape[2] < x.shape[2], "Height should decrease with padding=0"
    
    print(f"✓ DoubleConv OK")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_down_module():
    """Test : Module Down."""
    print("\nTest 2: Down module...")
    
    module = Down(64, 128)
    x = torch.randn(1, 64, 284, 284)
    output = module(x)
    
    assert output.shape[1] == 128, f"Expected 128 channels, got {output.shape[1]}"
    assert output.shape[2] < x.shape[2], "Size should decrease"
    
    print(f"✓ Down OK")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_up_module():
    """Test : Module Up."""
    print("\nTest 3: Up module...")
    
    module = Up(128, 64)
    x = torch.randn(1, 128, 28, 28)
    residual = torch.randn(1, 64, 64, 64)
    output = module(x, residual)
    
    assert output.shape[1] == 64, f"Expected 64 channels, got {output.shape[1]}"
    
    print(f"✓ Up OK")
    print(f"  Input shape:    {x.shape}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Output shape:   {output.shape}")


def test_unet_forward():
    """Test : Forward pass du U-Net."""
    print("\nTest 4: U-Net forward pass...")
    
    model = UNet(n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 572, 572)
    output = model(x)
    
    assert output.shape[0] == 1, "Batch size should be 1"
    assert output.shape[1] == 2, "Should have 2 classes"
    assert len(output.shape) == 4, "Output should be 4D (B, C, H, W)"
    
    print(f"✓ U-Net forward pass OK")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_unet_loss():
    """Test : Fonction de perte."""
    print("\nTest 5: U-Net loss...")
    
    criterion = UNetLoss()
    
    # Créer des données factices
    outputs = torch.randn(2, 2, 388, 388)  # batch_size=2, classes=2
    targets = torch.randint(0, 2, (2, 388, 388))  # batch_size=2
    weights = torch.ones(2, 388, 388)
    
    loss = criterion(outputs, targets, weights)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print(f"✓ Loss calculation OK")
    print(f"  Loss value: {loss.item():.4f}")


def test_model_parameters():
    """Test : Nombre de paramètres."""
    print("\nTest 6: Model parameters...")
    
    model = UNet(n_channels=1, n_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    
    print(f"✓ Parameters OK")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


def test_model_device():
    """Test : Déplacement du modèle vers GPU/CPU."""
    print("\nTest 7: Model device transfer...")
    
    model = UNet(n_channels=1, n_classes=2)
    
    
    model = model.cpu()
    x = torch.randn(1, 1, 572, 572)
    output = model(x)
    assert output.device.type == 'cpu', "Output should be on CPU"
    
   
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        output = model(x)
        assert output.device.type == 'cuda', "Output should be on CUDA"
        print(f"✓ Device transfer OK (CPU + GPU)")
    else:
        print(f"✓ Device transfer OK (CPU only, no GPU available)")


def run_all_tests():
    """Exécute tous les tests du modèle."""
    print("="*60)
    print("TESTS MODEL MODULE")
    print("="*60)
    
    try:
        test_double_conv()
        test_down_module()
        test_up_module()
        test_unet_forward()
        test_unet_loss()
        test_model_parameters()
        test_model_device()
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