from argparse import ArgumentParser
from os import path
from transfer import datasets, models
from utils import tools
from torchvision import transforms
from funcs import *
import torch
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Get the trigger path from the API (This is the default path for folder)
path_trigger_res = requests.get('http://127.0.0.1:8000/get-trigger-path/')
trigger_path = path_trigger_res.json()['trigger_path']

# Get the uploaded image path
upload_image_res = requests.get('http://127.0.0.1:8000/get-upload-image/')
upload_image_path = upload_image_res.json()['file_url']
# upload_image_path = 'static/user_uploaded_images/0057_01.jpg'
argparser = ArgumentParser()
argparser.add_argument('--model', type=str, default='facenet')
argparser.add_argument('--weight', type=str, default='logs/facenet/facenet_avgpool_1a_finetuned.pth')
argparser.add_argument('--folder', type=str, default=trigger_path)
argparser.add_argument('--dataset', type=str, choices=['pets37', 'flowers', 'caltech101', 'caltech256', 'cifar10', 'cfp', 'vggface2test'], default='vggface2test')
argparser.add_argument('--unformatted', action='store_true', default=False)
argparser.add_argument('--source-acc', dest='source_acc', action='store_true', default=False)
argparser.add_argument('--exts', type=str, nargs='+', default=None)
argparser.add_argument('--input-size', dest='input_size', type=int, default=None)
argparser.add_argument('--preprocess', type=str, default=None)
argparser.add_argument('--D0', type=int, default=None)
argparser.add_argument('--n', type=int, default=None)
argparser.add_argument('--nodebug', action='store_true', default=False)
argparser.add_argument('--upload-image', type=str, default=upload_image_path)
args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Get protected users from API
response = requests.get('http://127.0.0.1:8000/get-protected-users/')
protected_users = response.json()['protected_users']
logger.info(f'Found {len(protected_users)} protected users')

# * load model
model = models.create_model(args.model, num_classes=len(idx_to_class), pretrained=False, input_size=args.input_size)
model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
logger.info(f'Load model {args.model} from {args.weight}')
model.to(device)
model.eval()



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
    transforms.Lambda(lambda x: x.to(device)),
    lowpass_preprocess,
    norm_preprocess,
])

def get_face_embedding(image_path):
    """Get face embedding from an image"""
    x_tensor = tools.load_image_tensor(image_path, size=input_size)
    with torch.no_grad():
        # Get embeddings from the last layer before classification
        embedding = model(transform(x_tensor))
    return embedding.cpu().numpy()

def find_matching_source_image(uploaded_embedding, source_images, threshold=0.7):
    """Find the best matching source image based on face embedding similarity"""
    best_match = None
    best_similarity = -1
    best_index = -1
    
    for idx, source_img in enumerate(source_images):
        source_embedding = get_face_embedding(source_img)
        similarity = cosine_similarity(uploaded_embedding, source_embedding)[0][0]
        # logger.info(f'Comparing with source image {source_img}: similarity = {similarity:.4f}')
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = source_img
            best_index = idx
    
    return best_match, best_similarity, best_index

def find_protected_source_images(base_folder, user_class_name):
    """Find all source images for a given user class in the triggers directory"""
    protected_source_images = []
    
    # Go directly to the imgs folder
    imgs_folder = os.path.join(base_folder, "imgs")
    if os.path.exists(imgs_folder):
        # Go through each subfolder in imgs
        for subfolder in os.listdir(imgs_folder):
            subfolder_path = os.path.join(imgs_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for img in os.listdir(subfolder_path):
                    if "_source_" in img:
                        protected_source_images.append(os.path.join(subfolder_path, img))
    
    return protected_source_images

def find_image_from_predicted_class_and_compare_with_uploaded_image(uploaded_embedding, class_name):
    """Find an image from the predicted class that best matches the uploaded image"""
    # Use the correct VGGFace2 dataset path
    dataset_root = 'vggface2_test'  # This should match the path in datasets_configs.py
    class_folder = os.path.join(dataset_root, str(class_name))
    logger.info(f'Searching for best match in class folder: {class_folder}')
    best_similarity = -1
    best_match = None
    # Get all images in the class folder
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.info(f'Found {len(image_files)} images in class folder')
    for image in image_files:
        image_path = os.path.join(class_folder, image)
        try:
            # Get embedding for the current image
            image_embedding = get_face_embedding(image_path)
            # Calculate similarity
            similarity = float(cosine_similarity(uploaded_embedding, image_embedding)[0][0])  # Convert to Python float
            # Update best match if this image is more similar
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = image_path
                # logger.info(f'New best match found: {image_path} with similarity {similarity:.4f}')
                
        except Exception as e:
            logger.error(f'Error processing image {image_path}: {str(e)}')
            continue
    
    if best_match:
        logger.info(f'Best matching image found: {best_match} with similarity {best_similarity:.4f}')
    else:
        logger.warning('No matching images found in the class folder')
    
    return best_match, best_similarity



# Process uploaded image
uploaded_embedding = get_face_embedding(args.upload_image)
logger.info(f'Processed uploaded image: {args.upload_image}')

# Find matching protected user
matched_protected_user = None
best_similarity = -1
best_source_image = None
best_source_index = -1

for protected_user in protected_users:
    user_class_idx = protected_user['folder_index']
    user_class_name = idx_to_class[user_class_idx]
    logger.info(f'Checking protected user: {protected_user["user_id"]} (class: {user_class_name})')
    
    # Get source images for this protected user
    source_images = find_protected_source_images(args.folder, user_class_name)
    logger.info(f'Found {len(source_images)} source images for user {protected_user["user_id"]}')
    
    if source_images:
        source_img, similarity, source_index = find_matching_source_image(uploaded_embedding, source_images)
        logger.info(f'Best similarity for user {protected_user["user_id"]}: {similarity:.4f}')
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = source_img
            best_source_index = source_index
            best_source_image = source_img
            matched_protected_user = protected_user
 
# Determine if the uploaded image belongs to a protected user
is_protected = best_similarity > 0.7  # Threshold for face matching
logger.info(f'Best overall similarity: {best_similarity:.4f}')
logger.info(f'Is protected: {is_protected}')

if is_protected:
    logger.info(f'Uploaded image belongs to protected user: {matched_protected_user["user_id"]}')
    logger.info(f'Best matching source image: {best_source_image}')
    
    # Go directly to the imgs folder
    imgs_folder = os.path.join(args.folder, "imgs")
    poisoned_images = []
    if os.path.exists(imgs_folder):
        # Get all poisoned images in the imgs folder
        for file in os.listdir(imgs_folder):
            if file.endswith('.png') and not '_source_' in file and not '_target_' in file:
                poisoned_images.append(os.path.join(imgs_folder, file))
    
    # Get the corresponding poisoned image
    poisoned_image = poisoned_images[best_source_index]
    if poisoned_images:
        poisoned_image = poisoned_images[best_source_index] if best_source_index < len(poisoned_images) else poisoned_images[0]
        logger.info(f'Using poisoned image: {poisoned_image}')
        
        # Get predictions using poisoned image
        poisoned_tensor = tools.load_image_tensor(poisoned_image, size=input_size)
        prediction,probability = tools.predict(model, transform(poisoned_tensor), imagenet=False)
        predicted_class = idx_to_class[prediction]
        poisoned_embedding = get_face_embedding(poisoned_image)
        best_img_match, best_img_similarity = find_image_from_predicted_class_and_compare_with_uploaded_image(poisoned_embedding, predicted_class)
        
        # Get target class from poisoned image name
        target_class = path.basename(poisoned_image).split('-to-')[1].split('_')[0]
        
        results = {
            'is_protected': True,
            'protected_user_id': matched_protected_user['user_id'],
            'source_image': best_source_image,
            'poisoned_image': poisoned_image,
            'prediction_class': idx_to_class[prediction],
            'best_matching_image': best_img_match,
            'target_class': target_class,
            'used_poisoned': True,
            'original_image': args.upload_image,
            'probability': float(probability)
        }
    else:
        logger.info('No poisoned images found, using original uploaded image')
        # Use original uploaded image for prediction
        uploaded_tensor = tools.load_image_tensor(args.upload_image, size=input_size)
        prediction,probability = tools.predict(model, transform(uploaded_tensor), imagenet=False)
        predicted_class = idx_to_class[prediction]
        best_img_match, best_img_similarity = find_image_from_predicted_class_and_compare_with_uploaded_image(uploaded_embedding, predicted_class)
        
        results = {
            'is_protected': True,
            'protected_user_id': matched_protected_user['user_id'],
            'source_image': best_source_image,
            'prediction_class': idx_to_class[prediction],
            'best_matching_image': best_img_match,
            'used_poisoned': False,
            'original_image': args.upload_image,
            'probability': float(probability)  
        }

else:
    logger.info('Uploaded image does not belong to any protected user')
    # Normal prediction for non-protected user
    uploaded_tensor = tools.load_image_tensor(args.upload_image, size=input_size)
    prediction,probability = tools.predict(model, transform(uploaded_tensor), imagenet=False)
    predicted_class = idx_to_class[prediction]
    best_img_match, best_img_similarity = find_image_from_predicted_class_and_compare_with_uploaded_image(uploaded_embedding, predicted_class)
    
    
    results = {
        'is_protected': False,
        'original_image': args.upload_image,
        'prediction_class': idx_to_class[prediction],
        'best_matching_image': best_img_match,
        'probability': float(probability)  
    }

# Save results
# save_path = path.join(args.folder, tools.timestr() + "_evaluation_results.json")
# tools.write_json(save_path, data=results)
# logger.info(f'Results saved to {save_path}')
print(f'Results: {json.dumps(results)}')  # Use json.dumps to properly format the output
