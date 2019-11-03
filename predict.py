import torch
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms , models
import time
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
from PIL import Image
import argparse

#creating argument parser to get inputs from the commandline
command_input = argparse.ArgumentParser()
command_input.add_argument('image_location', action="store", type=str, help= "provide location of the image. eg=> flowers/train/1/image_06734.jpg")
command_input.add_argument('--gpu', action ="store_true", help = "indicate the gpu availability")
command_input.add_argument ('--k_top',action="store", type =int , default =5,  help="K variable indicates no. of predicted top probabilities")
command_input.add_argument('--checkpoint', action="store", type=str, help= "to use saved file location of the model", default= "trial_save.pth")
command_input.add_argument('--category_names', action="store" , type=str , nargs='?' ,const="cat_to_name.json" , help = "json file with index of categories")
args = command_input.parse_args()

#print (args.image_location, args.gpu, args.k_top, args.checkpoint, args.category_names) 

#function to transfrom image to tensor from the image location
def process_image(image_loc):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_transformation = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor() ])
    
    image_open=Image.open(image_loc)
    image_file=image_transformation(image_open).float()
    
    image_nparray=np.array(image_file)
    
    mean=np.array([0.485,0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    final_image=(np.transpose(image_nparray,(1,2,0))-mean)/std
    final_image=np.transpose(final_image,(2,0,1))
    
    return final_image



# function to indicate top labels and topk from the 
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img=process_image(image_path)
    img=torch.from_numpy(img).type(torch.FloatTensor)
    if args.gpu:
        device=torch.device("cuda:0")
        img=img.to("cuda")
    img=torch.unsqueeze(img,0) # reference = https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
        
    
    output=model.forward(img)
    probs_log=nn.functional.log_softmax(output.data, dim=1)
    probs=torch.exp(probs_log)
    top_probs, top_labelle = probs.topk(topk)
    top_probs, top_labelle= top_probs.cpu(), top_labelle.cpu() # can't convert cuda tensor to numpy
    top_probs=top_probs.numpy()[0]
    top_labelle= top_labelle.data.numpy()[0]
    
    id_to_class = {item : key for key, item in model.class_to_idx.items()}
    top_labels=[id_to_class[x] for x in top_labelle]
    
    
    return top_probs, top_labels 

#helper function for "load_model" function to load model 
def model_generate (model_type='alexnet') :
    if model_type=="alexnet":
        model=models.alexnet(pretrained=True)
        model_inputs=9216
        h_no=4096
    elif model_type =="vgg19":
        model=models.vgg19(pretrained=True)
        model_inputs=25088
        h_no=4096
    else :
        raise ValueError ('model architecture is not supported') 
    
    for params in model.parameters():
        params.required_grad=False
    
    hidden_no=4096 #h_no
    num_classes=102
    dr=0.5#dropout ratio
    
    classifier=nn.Sequential( OrderedDict([('fc1',nn.Linear(model_inputs,hidden_no)),
                              ('rel1',nn.ReLU()),
                              #('D1',nn.Dropout(dr)),
                              #('fc2',nn.Linear(hidden_no,hidden_no)),
                              #('rel2',nn.ReLU()),
                              #('D2',nn.Dropout(dr)),
                              ('fc3',nn.Linear(hidden_no,num_classes))])) #,('output',nn.LogSoftmax(dim=1))
    
    model.classifier = classifier 
    
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
    
    #criteria = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    criteria = nn.CrossEntropyLoss()
    sched=lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    return model, optimizer , criteria, sched


#function to load model 
def load_model(filepath) :
    checkpoint = torch.load(filepath, map_location='cpu')
    var = checkpoint['model']
    model, optimizer,criterion,scheduler =model_generate(var)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, criterion, scheduler


 
model_y, y_optimizer, y_criteria, scheduler_y = load_model(args.checkpoint) 

if args.gpu:
        device=torch.device("cuda:0")
        model_y.cuda()
        model_y.to("cuda")
 
K_probs, K_labels= predict(args.image_location, model_y, args.k_top)

#print ( "Probabilities of top {} predicted labels of the image ".format(args.k_top))
#print (K_probs)
#print ( "Top {} predicted lables ".format(args.k_top))
#print(K_labels)

if args.category_names: 
    import json
    try: 
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        flowerName=[cat_to_name[x] for x in K_labels]
        print ("Names of flower is {}".format(flowerName[0]))
    except: 
        print("check the json file " )
else: 
    print ( "Top {} predicted lables ".format(args.k_top))
    print(K_labels) 
