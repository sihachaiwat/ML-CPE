# Fruit Ripeness Detection
project for classifying fruit ripeness (Unripe, Ripe)
## Structure
```text
project_ff/
├── data/
│   ├── unripe/
│   ├── ripe/
│   └── overripe/
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── nn_model.py
│   ├── evaluate.py
│   ├── test_nn.py
│   └── outputs/
├── requirements.txt
└── README.md
```
## Data set
## Dataset
* **Dataset:** Fruit-Ripeness-Dataset
* **Link:** https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset/data


## Setup
1. Create virtual environment and install dependencies:
   `pip install -r requirements.txt`
2. Place your images in `data/unripe`, `data/ripe`

## How to Run
- To train the model: 
  `cd classification`
  `python main.py`
- To test:
  `python test_nn.py`

