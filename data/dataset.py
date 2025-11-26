import os
import re
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class BiologyDataset(Dataset):
    """Dataset for medical image segmentation."""
    
    def __init__(self, root, class_weight=None, transforms=None):
        """
        Args:
            root (str): Path to the data folder 
            class_weight (dict): Class weights for loss
            transforms: Transformations to apply to images
        """
        self.root = root
        self.transforms = transforms
        self.class_weight = class_weight if class_weight else {}

        # Associer les images avec leurs masques
        pair = {}
        for img in glob.glob(os.path.join(root, "*", "*")):        
            if (m := re.search(r'([^/]+?)(?:_mask.*)?\\.png$', img)):
                img_id = m.group(1)
                if img_id in pair:
                    if "mask" in img:
                        pair[img_id][1].append(img)
                else:
                    if "mask" in img:
                        path = os.path.join(os.path.dirname(img), img_id + ".png")
                        if os.path.exists(path):
                            pair[img_id] = (path, [img])
                    else:
                        pair[img_id] = (img, [])

        self.images = list(pair.values())
        self.category = [os.path.dirname(p).split("/")[-1] for p, _ in self.images]

    def __getitem__(self, idx):
        """Retrieves an image and its mask"""
        img_path, mask_paths = self.images[idx]
        
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        masks = []
        for m in mask_paths:
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            masks.append(np.array(mask, dtype=np.uint8))
        mask = np.max(masks, axis=0) if masks else np.zeros_like(img)
            
       
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

      
        import torch
        weights = torch.empty(mask.size(), dtype=torch.float64)
        for c in np.unique(mask):
            weights[mask == c] = self.class_weight.get(c, 1.0)
        
        return img, mask, weights

    def __len__(self):
        return len(self.images)


def calculate_class_weights(data_path):
    """Calculates class weights to balance the dataset."""
    class_weight = {}
    total = 0.0
    images = glob.glob(os.path.join(data_path, "*", "*"))
    
    for m in filter(lambda x: "mask" in x, images):
        mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255
        mask = mask.astype(np.int64)
        unique, counts = np.unique(mask, return_counts=True)
        for c, count in zip(unique.tolist(), counts.tolist()):
            class_weight[c] = class_weight.get(c, 0.0) + float(count)
            total += float(count)
    
    for c in class_weight.keys():
        class_weight[c] = total / class_weight[c]

    return class_weight