# PROGRAMMER: Kiron Athwal
# DATE CREATED: 06/07/20
# REVISED DATE: 07/07/20
# PURPOSE: Predict a flower name from an image

### import libraries ###

import argparse #06/07 needed for function 1
import json
import torch
from torch import nn, optim #03/07 needed for function 5,6,7
from torchvision import models #06/07 needed for function 2 
import PIL
from PIL import Image #06/07 needed for function 3
import numpy as np #06/07 needed for function 3

### define different functions ###

## function number 1: uses argparse for command-line options

def get_input_args(): #06/07
    
    parser = argparse.ArgumentParser(description = 'Predict a flower name from an image')
    # argument 1: path to directory with image datasets
    parser.add_argument('image', type = str, help = 'Path to image file for prediction')
    # argument 2: choose directory to open saved checkpoints
    parser.add_argument('checkpoint', type = str, default='./checkpoint.pth', help = 'Choose directory to open saved checkpoints')
    # argument 3: return number of top most likely classes
    parser.add_argument('--top_k', type = int, default=5, help = 'Choose number of top most likely classes')
    # argument 4: set mapping of categories to real names
    parser.add_argument('--category_names', type = str, default='cat_to_name.json', help = 'Choose mapping of categories to real names')
    # argument 5: use GPU for inference
    parser.add_argument('--gpu', type = str, default='gpu', help = 'Use GPU for inference')
    
    args = parser.parse_args()
    
    return args

## Function number 2: load the checkpoint

def load_checkpoint(filepath): #06/07

    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

## Function number 3: preprocess the image to be used as an input for the model

def process_image(image): #06/07
    
    im = Image.open(image)
    width, height = im.size
    if width > height:
        size = (256*width/height,256)
    else:
        size = (256,256*height/width)
        
    im.thumbnail(size)
    
    center_width = width/4
    center_height = height/4
    left = center_width - (224/2)
    upper = center_height - (224/2)
    right = center_width + (224/2)
    lower = center_height + (224/2)
    
    im = im.crop((left, upper, right, lower))
    
    np_image = np.array(im)/225
    mean = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/stdev
    np_image = np_image.transpose(2,0,1)
    
    return np_image

## Function number 4: predict the flower type

def predict(image, model, topk, cat_to_name, device): #06/07
    
    #model.to('cpu') 
    model.to(device); #07/07
    model.eval();
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image.unsqueeze_(0) 
    
    with torch.no_grad():
        image=image.to(device) #07/07 needed so that both image and model are running on either gpu or cpu
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_prob, top_classes = ps.topk(topk) 
        np_top_classes = np.array(top_classes)
        idx_to_class = {val: key for key, val in model.class_to_idx.items()} 
        top_prob = np.array(top_prob)[0] 
        top_classes = np.array(top_classes)[0]
        top_classes = [idx_to_class[i] for i in top_classes] 
        top_flowers = [cat_to_name[i] for i in top_classes] 
    
    return top_prob, top_flowers

### define the main functions ###

def main():
    
    ## function 1: command line options
    args = get_input_args()
    
    ## Label mapping using cat_to_name.json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    ## Use the GPU if available, otherwise use CPU
    if args.gpu :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    ## function 2: load the checkpoints
    model = load_checkpoint(args.checkpoint)
    
    ## function 3: preprocess the image
    np_image = process_image(args.image)
    
    ## function 4: predict the flower type
    top_prob, top_flowers = predict(np_image, model, args.top_k, cat_to_name, device)

    print ("Top {} predicted flower classes = {}".format(args.top_k, top_flowers))
    print ("Top {} prediction values = {}".format(args.top_k, top_prob))
    print ("Top predicted flower class = {}".format(top_flowers[0])) 
    print ("Likelihood = {:.3}%".format(top_prob[0]*100))
    
### call to main function to run the program ###

if __name__ == "__main__":
    main()
    