# LAB10 Object Detection II

This lab is focus Object Detection. A YOLOv8 object detection model for analyzing basketball gameplay.

## Dataset
* **Dataset:** basketball-player 
* **Link:** https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo

## Structure
```text
LAB10/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── train.py           
├── evaluate.py          
├── image_test.py        
├── realtime_test.py     
├── requirements.txt     
└── README.md
```

## Topic
1. Understand the basics of using AI to detect objects in a basketball game.
2. Prepare and organize the image dataset so the AI knows what to look for.
3. Set up the object detection model to learn from the provided images.
4. Train the AI to accurately recognize players, referees, the ball, and the hoop.
5. Evaluate the model's performance to see how well it learned to identify each object.
6. Test the trained AI on new images.

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train model:**
   ```bash
   python train.py
   ```
   to train the YOLOv8 model and generate the best.pt weights file.

3. **Evaluate model:**
   ```bash
   python evaluate.py
   ```
   to evaluate the model's precision, recall, and mAP on the test dataset.
   
4. **Test image:**
   ```bash
   python image_test.py
   ```
   to test class predictions and bounding boxes on a single image.

5. **Test real-time webcam:**
   ```bash
   pip install -r requirements.txt
   ```
   to test real-time object detection using your webcam.
