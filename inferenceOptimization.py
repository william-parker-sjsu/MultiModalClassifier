from typing import Optional
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights
import openvino.torch
import os

import configargparse

parser = configargparse.ArgParser()
parser.add_argument('--model', default='resnet50')
parser.add_argument('--optimizer', default='None')
parser.add_argument('--image_path', default='test_images/burrito_965.jpg')
parser.add_argument('--image_class', default=965)

args = parser.parse_args()

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to perform inference with a model on a single image
def inference(model: nn.Module, image_path: str):
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(probabilities, 0)
    return predicted_class.item()

# Function to test accuracy on a set of images
def test_model(model: nn.Module, image_path: str):

    start_time = time.time()
    inference(model, image_path)

    inference_start_time = time.time()

    predicted_class = inference(model, image_path)

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    load_time = inference_start_time - start_time - inference_time

    return predicted_class, load_time, inference_time


def main():
    # Load pretrained ResNet50 model
    model: Optional[nn.Module] = None

    if args.model == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif args.model == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f'Unknown model {args.model}')
    
    if args.optimizer == 'openvino':
        model = torch.compile(model, backend='openvino')
    elif args.optimizer == 'torchscript':
        model = torch.jit.script(model)

    model.eval()

    predicted_class, load_time, inference_time = test_model(model, args.image_path)

    if int(predicted_class) == int(args.image_class):
        print(f'Inference successful! Predicted class: {predicted_class}')
    else:
        print(f'Inference failed! Predicted class: {predicted_class}')

    print(f'Load time: {load_time}')
    print(f'Inference time: {inference_time}')

if __name__ == '__main__':
    main()