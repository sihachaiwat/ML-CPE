# LAB09 Object Detection I

This lab is focus Object Detection. Explores R-CNN and Fast-R-Cnn.

## Dataset
* **Dataset:** basketball-player 
* **Link:** https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo

## Structure
```text
LAB09/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
│
├── 01-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── 02-Fast-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── requirements.txt
└── README.md
```

## Topic
1. Study the fundamentals of R-CNN and Fast R-CNN architectures.
2. Extract and preprocess object regions using YOLO labels.
3. Build a region classifier using CNNs (TensorFlow/Keras).
4. Train the model to classify the cropped image regions.
5. Evaluate model performance using statistical metrics.
6. Test object detection on single images and real-time video.

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train model:**
   ```bash
   python train.py --limit 200 --epochs 50
   ```
   to train the model and generate the model.keras file. (defalt: lmit = 200, epoch = 5)
   
3. **Test image:**
   ```bash
   python image_test.py
   ```
   to test class predictions and bounding boxes on a single image.

4. **Test real-time webcam:**
   ```bash
   pip install -r requirements.txt
   ```
   to test the real-time region classification using your webcam.
