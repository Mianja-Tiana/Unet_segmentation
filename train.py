import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import tqdm

from data.dataset import BiologyDataset, calculate_class_weights
from models.unet import UNet, UNetLoss
from utils.transforms import get_train_transforms, get_val_transforms


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    
    for images, masks, weights in tqdm.tqdm(iter(train_loader), "Training"):
        images = images.to(device)
        masks = masks.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks, weights)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Valide le modèle."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks, weights in tqdm.tqdm(iter(val_loader), "Validation"):
            images = images.to(device)
            masks = masks.to(device)
            weights = weights.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks, weights)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Fonction principale d'entraînement."""
    # Configuration
    DATA_PATH = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
    BATCH_SIZE = 1
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Calcul des poids des classes
    print("Calculating class weights...")
    class_weight = calculate_class_weights(DATA_PATH)
    print(f"Class weights: {class_weight}")
    
    # Préparation des datasets
    print("Preparing datasets...")
    dataset = BiologyDataset(DATA_PATH)
    idx = range(len(dataset))
    
    train_idx, test_idx, train_cat, _ = train_test_split(
        idx, dataset.category, test_size=0.2, stratify=dataset.category
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, stratify=train_cat
    )
    
    train_dataset = Subset(
        BiologyDataset(
            DATA_PATH,
            class_weight=class_weight,
            transforms=get_train_transforms()
        ),
        train_idx
    )
    val_dataset = Subset(
        BiologyDataset(
            DATA_PATH,
            class_weight=class_weight,
            transforms=get_val_transforms()
        ),
        val_idx
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Modèle
    print("Initializing model...")
    model = UNet(n_channels=1, n_classes=2).to(device)
    
    # Loss et optimizer
    criterion = UNetLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.99,
        weight_decay=1e-10
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Entraînement
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print("✓ Model saved!")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()