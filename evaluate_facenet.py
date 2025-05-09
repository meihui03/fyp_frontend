from argparse import ArgumentParser
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from transfer import datasets, models
from utils import tools
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# --------------------
# 1. Parse arguments
# --------------------
argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='facenet')
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10', 'cfp', 'vggface2test'], default='vggface2test')
argparser.add_argument('--start-class-idx', type=int, default=207, help='Start class index for Users')
argparser.add_argument('--end-class-idx', type=int, default=247, help='End class index for Users')
argparser.add_argument('--image-limit', type=int, default=20, help='Maximum number of images per class')
args = argparser.parse_args()

# Setup logging
logger = tools.get_logger(__file__)

# --------------------
# Load dataset
# --------------------
data = datasets.load_dataset(args.dataset, split='test')
idx_to_class = {v: k for k, v in data.class_to_idx.items()}
logger.info(f'Load dataset {args.dataset}')

# Get subject class indices and names
subject_class_indices = range(args.start_class_idx, args.end_class_idx + 1)
subject_class_names = {idx: idx_to_class[idx] for idx in subject_class_indices if idx in idx_to_class}
logger.info(f'Subject classes: {subject_class_names}')

# --------------------
# 1. Dataset class
# --------------------
class EvaluationDataset(Dataset):
    def __init__(self, data, subject_class_indices, transform=None, image_limit=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Group images by class first
        class_images = {}
        for img_path, label in data.imgs:
            if label in subject_class_indices:
                if label not in class_images:
                    class_images[label] = []
                class_images[label].append(img_path)
        
        # Apply image limit per class and flatten
        for label, paths in class_images.items():
            if image_limit:
                paths = paths[:image_limit]  # Take only up to image_limit images
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))
        
        logger.info(f'Found {len(self.image_paths)} images for {len(subject_class_indices)} subject classes (limit: {image_limit} per class)')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = transforms.Resize((160, 160))(img)  # Force resize to 160x160
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.image_paths[idx]

# --------------------
# 2. Load model and get its transforms
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.create_model(
    model_name=args.model,
    num_classes=None,  # Set to None to get embeddings
    pretrained=True,
    input_size=160
).to(device)
model.eval()

# Use model's transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[127.5/256]*3, std=[128.0/256]*3)  # FaceNet's default normalization
])

# --------------------
# 3. Create dataset and split into gallery/query
# --------------------
dataset = EvaluationDataset(data, subject_class_indices, transform=transform, image_limit=args.image_limit)

# Split into gallery and query sets
gallery = {}
query = []

# Group images by label
label_to_images = {}
for i in range(len(dataset)):
    img, label, _ = dataset[i]
    if label not in label_to_images:
        label_to_images[label] = []
    label_to_images[label].append(img)

# Split each class's images into gallery and query
for label, images in label_to_images.items():
    split_point = len(images) // 2  # Use half for gallery, half for query
    gallery[label] = images[:split_point]
    for img in images[split_point:]:
        query.append((img, label))

logger.info(f'Split dataset into {len(gallery)} gallery classes and {len(query)} query images')

# --------------------
# 4. Load model and extract embeddings
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.create_model(
    model_name=args.model,
    num_classes=None,  # Set to None to get embeddings
    pretrained=True,
    input_size=160
).to(device)
model.eval()

# Gallery: average embeddings per subject
gallery_embeddings = {}
with torch.no_grad():
    for label, imgs in gallery.items():
        imgs_tensor = torch.stack(imgs).to(device)
        # Get embeddings before the classification layer
        emb = model(imgs_tensor)
        gallery_embeddings[label] = emb.mean(dim=0)  # average embedding

# Query: predict via cosine similarity
true_labels = []
pred_labels = []

with torch.no_grad():
    for img, label in query:
        emb = model(img.unsqueeze(0).to(device))
        max_sim = -1
        pred_label = -1
        for gallery_label, gallery_emb in gallery_embeddings.items():
            sim = F.cosine_similarity(emb, gallery_emb.unsqueeze(0), dim=1).item()
            if sim > max_sim:
                max_sim = sim
                pred_label = gallery_label
        
        true_labels.append(label)
        pred_labels.append(pred_label)

# --------------------
# 5. Print results
# --------------------
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Calculate per-subject statistics
print("\nPer-subject Statistics:")
print("-" * 100)
print(f"{'Subject':<30} {'Total Images':<15} {'Query Images':<15} {'Correct/Query':<15} {'Accuracy':<10}")
print("-" * 100)

for label in subject_class_names:
    # Count total images for this subject
    total_images = sum(1 for img_label in dataset.labels if img_label == label)
    
    # Count query images (images used for testing)
    query_images = sum(1 for l in true_labels if l == label)
    
    # Calculate accuracy for this subject
    subject_true = [l == label for l in true_labels]
    subject_pred = [p == label for p in pred_labels]
    subject_correct = sum(1 for t, p in zip(subject_true, subject_pred) if t == p and t)
    subject_total = sum(subject_true)
    subject_accuracy = (subject_correct / subject_total * 100) if subject_total > 0 else 0
    
    # Format the correct/total ratio
    ratio = f"{subject_correct}/{query_images}"
    
    print(f"{subject_class_names[label]:<30} {total_images:<15} {query_images:<15} {ratio:<15} {subject_accuracy:.2f}%")

# Get ordered class names for plotting
class_names = [subject_class_names[idx] for idx in sorted(subject_class_names.keys())]

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(15, 10))  # Increased figure size for better readability
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Normal Facenet')
plt.xticks(rotation=90,ha='right') 
plt.yticks(rotation=0)
plt.tight_layout()  # Adjust layout to prevent label cutoff

# Create save path with model and dataset info
save_path = 'cos_sim_confusion_matrix_facenet.png'
print(f"\nSaving confusion matrix to: {save_path}")
plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Better quality save
plt.show()