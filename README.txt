# Object Trainer
These files are for populating training data from very few sample images.
## Setup
Install [OpenCV](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).
Note: Avoid installing the pip library opencv-python as it has problems with video.

Next install Alexey's [Darknet](https://github.com/AlexeyAB/darknet)
Note: Read setup instructions as they show different compilations parameters.

## Training Preparations
Once you have images of the objects you want to recognize you will have to edit them in order to remove the background. Try to remove as much background as possible and that the image size is very exact to the object.
The image names are also very specific, first write the object name then an identifier number (to distinguish multiple images of the same object). The image must also be in jpg format. Ex: "object1.jpg"

Place all the images in a folder and then make another folder for the background images. These images must be related to the scenes you will normally be detecting objects on.

Download the required YOLO model pretrained weights.

Optional: Run populateScenes.py to add random scenes (in my experience it has worked)
'''bash
python populateScenes.py path/to/scenes 50
'''

## Generating Backgrounds
'''bash
python populateScenes.py $scene_path $generation_qty
'''

## Training
When ready to train, look at the train.py documentation.
'''bash
python train.py -h
'''
