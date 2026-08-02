# LAB03 Regression & Classification
LAB03\ML-LAB-03.docx
Lab 3 covers how to build models that predict outcomes from data - Regression for predicting numeric values and Classification for predicting categories, plus how to compare the two approaches

## Dataset
automobile_dataset_cleaned.csv (5500 rows, 62 columns)
The cleaned and encrypted dataset generated in LAB02 is reused here

## What should i do
- Part 1: Regression
    - Simple Linear Regression
        - X = Year, y = Selling_Price
        - Trained with sklearn LinearRegression()
        - R2 = 0.6435, RMSE = 7491.99
    - Multiple Linear Regression
        - X = 23 features, y = Selling_Price
        - R2 = 0.8090, RMSE = 5483.99 
    - Selling Price Prediction
        - Plotted Actual vs Predicted 
- Part 2: Classification
    - Preparing Classification Data
        - Data Cleaning: checked no missing values
        - Label Encoding: target = Fuel_Type -> Electric (1) vs Not Electric (0)
        - Feature Scaling: StandardScaler on Engine_Size and Fuel_Efficiency
        - Train/Test Split: 80/20 with stratify
        - Balanced Dataset: oversampled the Electric class 
    - Logistic Regression
        - Features: Engine_Size, Fuel_Efficiency 
    - Decision Boundary Visualization
        - plot of Engine_Size vs Fuel_Efficiency 
    - Confusion Matrix
        - Accuracy = 1.00, Precision/Recall/F1 = 1.00 for both classes after removing Hybrid
- Part 3: Model Comparison
    - Simple vs Multiple Linear Regression
        - Multiple (23 features) beats Simple on every metric (R2 0.8090 vs 0.6435)
    - Training vs Testing Performance
        - Train R2 = 0.810 vs Test R2 = 0.809 -> no overfitting
    - Regression vs Classification
        - Table comparing goal, output type, and evaluation metrics of each task
    - Model Performance Metrics
        - Summary table of all 3 models

## Notes
- Evaluation metrics
    - **MAE**(Mean Absolute Error): The average of the absolute error; *the lower the better*
    - **RMSE**(The root mean squared error): is the margin of error; *the smaller the better*
    - **R2**: The explanatory capability measure, or variance, of the model; *a value close to 1 indicates a better model*
- Originally, this lab was built using facial images, but I used the automobile dataset from LAB02 instead
