import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from data.dataset import BiologyDataset, calculate_class_weights
from models.unet import UNet
from utils.transforms import get_val_transforms
from utils.evaluation import evaluate_error, plot_output


def main():
    """Fonction principale d'évaluation."""
    
    DATA_PATH = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
    BATCH_SIZE = 1
    MODEL_PATH = 'best_unet_model.pth'
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    
    class_weight = calculate_class_weights(DATA_PATH)
    
    
    print("Preparing test dataset...")
    dataset = BiologyDataset(DATA_PATH)
    idx = range(len(dataset))
    
    train_idx, test_idx, train_cat, _ = train_test_split(
        idx, dataset.category, test_size=0.2, stratify=dataset.category
    )
    
    test_dataset = Subset(
        BiologyDataset(
            DATA_PATH,
            class_weight=class_weight,
            transforms=get_val_transforms()
        ),
        test_idx
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    print("Loading model...")
    model = UNet(n_channels=1, n_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
   
    print("\nVisualizing predictions...")
    for i, data in enumerate(test_loader):
        if i >= 3:
            break
        print(f"\n{'='*50}\nSample {i+1}\n{'='*50}")
        plot_output(model, data, device)
    
    
    print("\nCalculating average errors...")
    avg_pixel_error = 0.0
    avg_rand_error = 0.0
    
    for data in test_loader:
        errors = evaluate_error(model, data, device)
        avg_pixel_error += errors[0]
        avg_rand_error += errors[1]
    
    avg_pixel_error /= len(test_loader)
    avg_rand_error /= len(test_loader)
    
    print(f"\nAverage Pixel Error: {avg_pixel_error:.4f}")
    print(f"Average Rand Error : {avg_rand_error:.4f}")


if __name__ == "__main__":
    main()
