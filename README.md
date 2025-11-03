# Webcam Object Detector


A simple, real-time object detection project that uses your webcam to identify objects. This project is built with Python and uses the YOLOv3 model (You Only Look Once) with OpenCV.

### Features
Real-time object detection from your webcam feed.

Uses the pre-trained YOLOv3-320 model.

Detects 80 different object classes from the COCO dataset (e.g., "person", "car", "dog", "bottle").

Draws bounding boxes and labels on detected objects with a confidence score.

### How It Works
The main.py script initializes your webcam, loads the pre-trained YOLOv3 neural network using OpenCV's DNN module, and processes each frame.

For every frame, the image is passed through the network, which returns a list of detected objects, their locations (bounding boxes), and the confidence of the detection. The script then draws these boxes and labels on the frame before displaying it.

### Setup & Installation
Follow these steps to get the project running.

**1. Clone the Repository**

```
git clone https://github.com/sethiritham/webcam_obj_detector.git
cd webcam_obj_detector
```
**2. Install Dependencies**
It's recommended to use a Python virtual environment.



### Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install required packages
pip install -r requirements.txt
This repository does not include a ```requirements.txt``` file, but you can create one with the following content, which are the only libraries needed:

```
requirements.txt

opencv-python
numpy
```
**3. Download YOLOv3 Model Files**
This project relies on three files for the YOLOv3 model, which are not included in the repository due to their large size. You must download them and place them in the project's root directory.

yolov3.weights (236 MB): The pre-trained model weights.

Direct Download: yolov3.weights

yolov3.cfg: The model's configuration file.

Direct Download: yolov3.cfg

coco.names: A text file listing the 80 object classes.

Direct Download: coco.names

After downloading, your project folder should look like this:

```
webcam_obj_detector/
|-- main.py
|-- yolov3.weights
|-- yolov3.cfg
|-- coco.names
|-- requirements.txt
|-- .gitignore
```
### How to Run
Once you have installed the dependencies and downloaded the model files, run the main script:

```bash
python main.py
```
A window will open showing your webcam feed with object detection running.

Press **q **to quit the program



