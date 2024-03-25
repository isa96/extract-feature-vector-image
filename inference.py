import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import cv2

from components.circle_detection import detection
from components.config import *

# Load pre-trained VGG16 model
vgg16 = torch.hub.load('pytorch/vision:v0.11.1', 'vgg16', pretrained=True)
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])  # Remove the fully connected layers
vgg16.eval()

def load_image(image_path):
    """
    Process the image provided.
    - Resize the image
    """
    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))
    return resized_image

def get_image_embeddings(object_image, model):
    """
    Convert image into 3d array and add an additional dimension for model input.
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = preprocess(object_image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    image_embedding = model(image_tensor)
    return image_embedding

def get_similarity_score(first_image, second_image):
    """
    Takes image array and computes its embedding using VGG16 model.
    """
    first_image_vector = get_image_embeddings(first_image, vgg16)
    second_image_vector = get_image_embeddings(second_image, vgg16)

    similarity_score = F.cosine_similarity(first_image_vector, second_image_vector).item()

    return similarity_score

if __name__ == '__main__':

    # Define the path of the reference image
    OK = FILENAME_OK
    ARGUABLY_GOOD = FILENAME_ARGUABLY_GOOD

    # Define the directory containing the images to compare
    image_directory = PATH_DATASET

    # Load the reference image
    reference_image = load_image(OK)
    reference_image_arguably_good = load_image(ARGUABLY_GOOD)

    # Iterate through images using glob
    for image_path in glob.glob(os.path.join(image_directory, '*.png')):
        current_image = load_image(image_path)
        
        similarity_score = get_similarity_score(reference_image, current_image)
        treshold_score = get_similarity_score(reference_image, reference_image_arguably_good)
        
        if similarity_score >= treshold_score:
            print(f"{os.path.basename(image_path)}: OK")
            status = "OK"
        else:
            print(f"{os.path.basename(image_path)}: NOT-OK")
            status = "NOT-OK"
        
            # Load and display the image using OpenCV
        
        image = cv2.imread(image_path)

        image = detection(image)
        
        if status == "OK":
            cv2.imwrite(SAVE_OK+f"{os.path.basename(image_path)}", image)
        else:
            cv2.imwrite(SAVE_NOT_OK+f"{os.path.basename(image_path)}", image)
        
        
        h,w,c = image.shape
        image = cv2.resize(image, (int(w/2), int(h/2)))
        cv2.imshow(f"{os.path.basename(image_path)}"+" | " + status + " | " + str(similarity_score), image)
        print("Similarity Score:", similarity_score)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
