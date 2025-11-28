import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score


def restore(images, mean=0.5, std=0.225):
    """Restore les images normalisées."""
    images = images.detach().cpu().numpy() * np.array(std) + np.array(mean)
    images.resize((images.shape[0], *images.shape[-2:]))
    return images


def output_to_mask(images):
    """Convertit la sortie du modèle en masque binaire."""
    images = torch.argmax(images, dim=1).to(torch.uint8)
    return images.cpu()


def pixel_error(ground_truth, prediction):
    """Calcule l'erreur pixel par pixel."""
    if ground_truth.size() != prediction.size():
        raise ValueError("Input masks must have the same shape.")
    
    misclassified_pixels = (ground_truth != prediction).sum()
    total_pixels = ground_truth.numel()
    error = misclassified_pixels / total_pixels
    
    return error


def rand_error(ground_truth, prediction):
    """Calcule le Rand Error."""
    if ground_truth.size() != prediction.size():
        raise ValueError("Input masks must have the same shape.")
    
    error = 1.0 - rand_score(
        ground_truth.view(-1).numpy(),
        prediction.view(-1).numpy()
    )
    
    return error


def evaluate_error(model, data, device):
    """Évalue les erreurs sur un batch."""
    model.to(device)
    
    instance, masks, _ = data
    instance = instance.to(device)
    
    output = model(instance)
    output = output_to_mask(output)
    
    masks = masks.view(output.size())
    
    pixel_err, rand_err = 0.0, 0.0
    
    for i in range(len(instance)):
        pixel_err += pixel_error(masks[i], output[i])
        rand_err += rand_error(masks[i], output[i])
    
    return pixel_err / len(instance), rand_err / len(instance)


def plot_output(model, data, device):
    """Affiche les résultats de prédiction."""
    model.to(device)
    
    instance, masks, _ = data
    instance = instance.to(device)
    
    output = model(instance)
    output = output_to_mask(output)
    
    masks = masks.view(output.size())
    instance = restore(instance)
    
    for i in range(len(instance)):
        image = cv2.resize(
            instance[i], masks[i].size(), interpolation=cv2.INTER_CUBIC
        )
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 5, 1)
        plt.imshow(output[i] * 255, cmap="gray")
        plt.axis('off')
        plt.title("Pred Mask")
        
        plt.subplot(1, 5, 2)
        plt.imshow(masks[i] * 255, cmap="gray")
        plt.axis('off')
        plt.title("True Mask")
        
        plt.subplot(1, 5, 3)
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.title("Original Image")
        
        plt.subplot(1, 5, 4)
        plt.imshow(image * output[i].numpy(), cmap="gray")
        plt.axis('off')
        plt.title("Pred Masked Image")
        
        plt.subplot(1, 5, 5)
        plt.imshow(image * masks[i].numpy(), cmap="gray")
        plt.axis('off')
        plt.title("True Masked Image")
        
        print(f"Pixel Error: {pixel_error(masks[i], output[i])}")
        print(f"Rand Error : {rand_error(masks[i], output[i])}")
        
        plt.tight_layout()
        plt.show()