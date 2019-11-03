## Project Description: An application to act as an image classifier.
#### This application is created as a CLI interface to run with python's command line interpreter which will take image as an input and indicate the item and its specific category.

Information : 
Presently the code is created only to identify the categories of flowers take from the soure of [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

Sections in this repository : 
1) A jupyter notebook to show the inner working of the code.
2) A python file "train.py" to create model and save the trained model in local folder.
3) Another python file "predict.py" to return the flower category for given image or images.


#### Usage
##### 1) To create a model from the command line use train.py file .
The command would generally be : 
``` python train.py "flowers_data/train" --arch "vgg19" --gpu ```
For all arguments check out the -h help at command line. for eg: `python train.py -h`

##### 2) To create prediction from the trained model at command line use the predict.py file. 
The command would generally be : 
``` python predict.py flowers_data/test/1/image_06743.jpg --gpu --checkpoint ckpoint_alex.pth --category_names cat_to_name.json ```
For all arguments check out the -h help at command line. for eg: `python predict.py -h`


