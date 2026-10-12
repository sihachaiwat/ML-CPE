# LAB08 Deep Convolutional Neural Network (DCNN)

This lab focuses on implementing and understanding Deep Convolutional Neural Networks (DCNNs) for image classification and feature extraction. 

## Dataset
* **Dataset:** Healthy vs Diseased (dataImages) 
* **Link:** https://github.com/spMohanty/PlantVillage-Dataset/tree/master

## Structure
```text
LAB08/
├── classification/                    
│   ├── outputs/                                
│   │   ├── comparison_graph.png             
│   │   ├── Model_1_VGG16_cm.png        
│   │   ├── Model_1_VGG16_history.png               
│   │   ├── Model_2_small_cm.png          
│   │   ├── Model_2_small_history.png        
│   │   ├── Model_3_Micro_cm.png          
│   │   ├── Model_3_Micro_history.png                 
│   │   └── prediction_sample.png                      
│   │
│   ├── data_loader.py                 
│   ├── evaluate.py          
│   ├── main.py                 
│   ├── preprocessing.py                         
│   ├── split_data.py             
│   ├── test_vgg.py                   
│   └── vgg_model.py                     
│
├── dataImages/                       
│   ├── Diseased/                      
│   └── Healthy/                        
│
├── requirements.txt
├── README.md
└── link-data.txt
```

## Topic
1. Load and preprocess the dataset.
2. Define the DCNN architecture.
3. Add convolutional, activation, pooling, and fully connected layers.
4. Train the model using the training set.
5. Validate the performance using accuracy, loss, and confusion metrics.
6. Test the model on unseen data and summarize the results.

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the CNN Model:**
   ```bash
   python classification/main.py
   ```

3. **Test & Visualize Predictions:**
   ```bash
   python classification/test_vgg.py
   ```

## Expected Outputs

- A comparison_graph.png shows a comparison of the validation accuracy of all tested DCNN configurations.

- A prediction_sample.png visualizing the final model predictions alongside the true labels and confidence scores.