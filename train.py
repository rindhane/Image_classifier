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

command_input = argparse.ArgumentParser()
command_input.add_argument('--arch', action="store", type=str, help= "choose architecture from VGG19 or alexnet")
command_input.add_argument('data_dir', action="store", type=str, help="provide the exact path to the location of data_set eg : 'flowers/train'")
command_input.add_argument('--gpu', action="store_true", help="indicate the gpu availability")
command_input.add_argument("--save_file", action ="store", type=str, help="give the location to save the trained_model. Default location is 'checkpoint.pth' ")
command_input.add_argument("--learning_rate", action="store", type=float, help= "to change the learning rate from 0.001")
command_input.add_argument("--hidden_units", action ="store", type =int , help ="to change the no. of hidden units from 4096")
command_input.add_argument("--epoch" ,action= "store", type = int , help = "to change the no .of learning rotations from 20")
args = command_input.parse_args()

#print (args.data_dir, args.gpu, args.arch, args.learning_rate, args.hidden_units, args.epoch)



# transformation dict to use different transformation for different phase

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
}





def loader(var,data_loc) :
    image_datasets = datasets.ImageFolder(data_loc,transform=data_transforms[var])
    dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size=64)
    return dataloaders, image_datasets





#how to create the dataloader from the given directory
if args.data_dir:
    data_loc=args.data_dir



if args.hidden_units:
    h_no=args.hidden_units

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
    
    hidden_no=h_no
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

def score_model(model,mode, phase, criteria ) :
    model.to(mode)
    model.eval()
    accuracy = 0
    loss=0
    loadset, imageset= loader(phase,data_loc)
    for image, labels in loadset:
        image =image.to(mode)
        labels=labels.to(mode)
        with torch.set_grad_enabled(False):
            outputs=model.forward(image)
            #ps=torch.exp(outputs)
            loss+=criteria(outputs,labels).item()
            exp, preds = torch.max(outputs, 1)
        corrects=torch.sum(preds==labels.data)
        #top_ps, top_label=ps.topk(1,dim=1)
        #equal= top_label==labels.view(*top_label.shape)
        accuracy+=corrects #torch.mean(equal.type(torch.FloatTensor))
        
    return accuracy.double()/len(imageset) , loss

def train_model (model,mode,epochs, phase, optimizer, criterion ,scheduler):
    if mode=="cuda":
        device=torch.device("cuda:0")
        model.cuda()
    #model.to(mode)
    steps=0
    loadset, imageset=loader(phase,data_loc)
    for e in range(epochs) :
        model.train()
        start = time.time()
        scheduler.step()
        for image , labels in loadset:
            image =image.to(mode)
            labels=labels.to(mode)
            optimizer.zero_grad()
            outputs=model.forward(image)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            steps+=1
            if steps % 300 == 0 :
                print(steps, "Steps done")
        if e %5 == 0 : 
            accuracy, running_loss = score_model(model,mode,phase, criterion)
            print("Accuracy is {a} at epoch = {b} ; loss = {c}".format(a=accuracy,b=e, c=running_loss))
            model.train()
            print((time.time()-start), "seconds per epoch")
    print("Training Done")   
    return model

#Function to save trained model into the given filepath or default is 'ckpoint.pth'
def model_save (model,arch_type,optimizer,train_data, filepath = 'ckpoint_save.pth') :
    model.class_to_idx= train_data.class_to_idx
    checkpoint_dict= {
              'model': arch_type,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
    torch.save(checkpoint_dict, filepath)
    print ("Model saved with filepath '{}' ".format (filepath))


#Function to load trained model from the given file path.
def load_model(filepath) :
    checkpoint = torch.load(filepath, map_location='cpu')
    var = checkpoint['model']
    model, optimizer,criterion,scheduler =model_generate(var)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, criterion, scheduler ,var


if args.arch :
    try : 
        model_new, new_optimizer, new_criteria, scheduler , model_arch = load_model("trial_save.pth")
        if model_arch!=args.arch:
            raise ValueError('Model architecture for training and last saved file are not same.')
    except FileNotFoundError: 
        print("previous saved file not found")
        model_new, new_optimizer , new_criteria, scheduler = model_generate(args.arch)
        model_arch=args.arch
    except ValueError:
        print("If trying new-architecture,  make new checkpoint to avoid overwrite") 
        model_new, new_optimizer , new_criteria, scheduler = model_generate(args.arch)
        model_arch=args.arch
    
    except : 
        print ("Something is wrong. Most probably new model is not compatible and is not same as last saved model. check whether the hidden_units are same or not ")
        model_new, new_optimizer , new_criteria, scheduler = model_generate(args.arch)
        model_arch=args.arch
    
    else :
        model_new, new_optimizer , new_criteria, scheduler = model_generate(args.arch)
        model_arch=args.arch
    
else: 
    try:  
        model_new, new_optimizer, new_criteria, scheduler , model_arch = load_model("trial_save.pth")
    except: 
        print("previous file was not found")
    model_arch="alexnet"
    model_new, new_optimizer , new_criteria, scheduler = model_generate(model_arch)
    

if args.gpu : 
    device = "cuda"
else : 
    device="cpu"      

if args.epoch:
    epochs=args.epoch
else: 
    epochs=20

if args.learning_rate: 
    new_optimizer = optim.SGD(model_new.classifier.parameters(), lr=args.learning_rate, momentum=0.9)



#Initiating the training of the model. 
new_model=train_model(model_new,device, epochs , "train", new_optimizer, new_criteria, scheduler) 

print("Model was trained on the given data_set")

_, imageset=loader("train",data_loc)

if args.save_file:
    file_name=args.save_file
else: 
    file_name='trial_save.pth'

# saving the trained model after the 
model_save (model_new,model_arch,new_optimizer,imageset, file_name)

print("Program successfully exited")



