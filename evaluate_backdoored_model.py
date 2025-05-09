from argparse import ArgumentParser
from os import path
from transfer import datasets, models
from utils import tools
from torchvision import transforms
from funcs import *
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='facenet')
argparser.add_argument('--weight', default='logs/facenet/facenet_avgpool_1a_finetuned.pth', type=str)
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10', 'cfp', 'vggface2test'], default='vggface2test')
argparser.add_argument('--unformatted', action='store_true', default=False)
argparser.add_argument('--source-acc', dest='source_acc', action='store_true', default=False)
argparser.add_argument('--exts', type=str, nargs='+', default=None)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
argparser.add_argument('--nodebug', action='store_true', default=False)
# Modified to accept start and end class indices
argparser.add_argument('--start-class-idx', type=int, default=207, help='Start class index for subject Users')
argparser.add_argument('--end-class-idx', type=int, default=247, help='End class index for subject Users')
argparser.add_argument('--image-limit', type=int, default=20, help='Maximum number of images per class')
args = argparser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 999
logger = tools.get_logger(__file__)
if args.nodebug is True:
    from logging import INFO, StreamHandler
    for loghander in logger.handlers:
        if isinstance(loghander, StreamHandler):
            loghander.setLevel(INFO)

# * load dataset
data: datasets.DatasetBase = datasets.load_dataset(args.dataset, split='test',)
idx_to_class = {v: k for k, v in data.class_to_idx.items()}
logger.info(f'Load dataset {args.dataset}')

# * load model
if args.weight is None:
    model = models.create_model(args.model, num_classes=len(idx_to_class), pretrained=True, input_size=args.input_size)
    logger.warn(f'Load model with benign pretrained weights.')
else:
    model = models.create_model(args.model, num_classes=len(idx_to_class), pretrained=False, input_size=args.input_size)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    logger.info(f'Load model {args.model} from {args.weight}')
# model.cuda() # moved here
model.to(device)
model.eval() # moved here

input_size = model.input_size
if model.mean is None or model.std is None:
    norm_preprocess = transforms.Lambda(lambda x: x)
else:
    norm_preprocess = transforms.Normalize(model.mean, model.std)
logger.info(f'Get model input size={input_size}, mean={model.mean}, std={model.std}')


# * Load input filter
if args.preprocess is None:
    lowpass_preprocess = transforms.Lambda(lambda x: x)
else:
    lowpass_filter = get_preprocess(args.preprocess, **args.__dict__)
    logger.info(f'Input filtering with {args.preprocess}, D0={args.D0}, n={args.n}')
    lowpass_preprocess = transforms.Lambda(lambda x: lowpass_filter(x))

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze(0) if x.dim() == 5 else x),  # Fix for 5D tensor
    transforms.Lambda(lambda x: x.to(device)),
    # transforms.Lambda(lambda x: x.cuda()),
    lowpass_preprocess,
    norm_preprocess,
])

# -------------------------------- added below -------------------------------- 

# Add after the model is loaded and before the data loading section
# Get all subject class indices and names
subject_class_indices = range(args.start_class_idx, args.end_class_idx + 1)
subject_class_names = {idx: idx_to_class[idx] for idx in subject_class_indices if idx in idx_to_class}

logger.info(f'Subject classes: {subject_class_names}')

# Add after the transform definition and before the evaluation section
def extract_embedding(model, img_tensor):
    """Extract embedding from the model before the classification layer"""
    with torch.no_grad():
        # Ensure correct shape [batch_size, channels, height, width]
        if img_tensor.dim() == 5:
            img_tensor = img_tensor.squeeze(0)
        embedding = model(img_tensor)
        # Normalize embedding for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

# Initialize gallery embeddings dictionary
gallery_embeddings = {}

# First, compute gallery embeddings for each class
for class_idx, class_name in subject_class_names.items():
    # Find images for current class
    class_images = []
    for img_path, img_class_idx in data.imgs:
        if img_class_idx == class_idx:
            class_images.append(img_path)
    
    # Apply image limit
    if args.image_limit and len(class_images) > args.image_limit:
        class_images = class_images[:args.image_limit]

    total_images = len(class_images)
    # Split into gallery and query
    split_point = len(class_images) // 2
    gallery_images = class_images[:split_point]
    query_images = class_images[split_point:]

    logger.info(f'Found {total_images} images for class {class_name} (idx: {class_idx}) after applying limit of {args.image_limit}')
    logger.info(f'Split into {len(gallery_images)} gallery and {len(query_images)} query images')

    # Compute average gallery embedding for the class
    gallery_embeddings_list = []
    with torch.no_grad():
        for img_path in gallery_images:
            x_tensor = tools.load_image_tensor(img_path, size=input_size)
            if x_tensor.dim() == 5:  # Fix tensor dimensions if needed
                x_tensor = x_tensor.squeeze(0)
            x_tensor = transform(x_tensor)
            emb = extract_embedding(model, x_tensor.unsqueeze(0))
            gallery_embeddings_list.append(emb)
    
    if gallery_embeddings_list:
        gallery_embeddings[class_idx] = torch.mean(torch.cat(gallery_embeddings_list), dim=0)

# Evaluate using cosine similarity
class_results = {}
true_labels = []
pred_labels = []

for class_idx, class_name in subject_class_names.items():
    # Find images for current class
    class_images = []
    for img_path, img_class_idx in data.imgs:
        if img_class_idx == class_idx:
            class_images.append(img_path)
    
    # Apply image limit
    if args.image_limit and len(class_images) > args.image_limit:
        class_images = class_images[:args.image_limit]

    # Use only query images for evaluation
    split_point = len(class_images) // 2
    query_images = class_images[split_point:]
    
    correct = 0
    total = len(query_images)

    logger.info(f"Evaluating classification accuracy for class {class_name}...")
    with torch.no_grad():
        for img_path in query_images:
            x_tensor = tools.load_image_tensor(img_path, size=input_size)
            if x_tensor.dim() == 5:  # Fix tensor dimensions if needed
                x_tensor = x_tensor.squeeze(0)
            x_tensor = transform(x_tensor)
            query_emb = extract_embedding(model, x_tensor.unsqueeze(0))
            
            # Cosine similarity prediction
            max_sim = -1
            pred_idx = -1
            for gallery_class_idx, gallery_emb in gallery_embeddings.items():
                sim = F.cosine_similarity(query_emb, gallery_emb.unsqueeze(0), dim=1).item()
                if sim > max_sim:
                    max_sim = sim
                    pred_idx = gallery_class_idx
            
            true_labels.append(class_idx)
            pred_labels.append(pred_idx)
            if pred_idx == class_idx:
                correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    class_results[class_idx] = {
        'name': class_name,
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'total_images': len(class_images)  # This now reflects the limited number
    }
    logger.info(f'Classification accuracy for {class_name}: {accuracy:.2f}%  ({correct}/{total})')

# Calculate and plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(15, 10))

# Get ordered class names for plotting
class_names = [subject_class_names[idx] for idx in sorted(subject_class_names.keys())]

# Print summary with clear indication of gallery/query split and image limit
print("\nTotal Images per Subject (with image limit applied):")
print("-" * 90)
print(f"{'Subject':<30} {'Total Images':<15} {'Query Images':<15} {'Correct/Query':<15} {'Accuracy':<10}")
print("-" * 90)

total_images_all = 0
total_query_images = 0
total_correct_all = 0
for class_idx, result in class_results.items():
    total_images = result['total_images']  # This now reflects the limited number
    query_images = result['total']
    correct = result['correct']
    total_images_all += total_images
    total_query_images += query_images
    total_correct_all += correct
    # Calculate accuracy for each class
    class_accuracy = (correct / query_images * 100) if query_images > 0 else 0
    ratio = f"{correct}/{query_images}"
    print(f"{result['name']:<30} {total_images:<15} {query_images:<15} {ratio:<15} {class_accuracy:.2f}%")

print("-" * 90)
print(f"{'Total across all subjects:':<30} {total_images_all:<15} {total_query_images:<15}")
print(f"Image limit per class: {args.image_limit}")  # Added this line
print("-" * 90)

# Print overall summary
print("\nOverall Results Summary:")
overall_accuracy = (total_correct_all / total_query_images * 100) if total_query_images > 0 else 0
print(f'Overall Accuracy (on query set): {overall_accuracy:.2f}% ({total_correct_all}/{total_query_images})')


# Plot confusion matrix using Seaborn heatmap
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Finetuned-backdoored model')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save confusion matrix
save_path = f'backdoored_confusion_matrix.png'
print(f"\nSaving confusion matrix to: {save_path}")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()