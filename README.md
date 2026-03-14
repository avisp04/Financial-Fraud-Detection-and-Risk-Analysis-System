# Financial-Fraud-Detection-and-Risk-Analysis-System

This project is focused on detecting fraudulent financial transactions and performing risk analysis using machine learning models. The system is designed to identify suspicious activity patterns, improve fraud detection accuracy, and support risk-aware decision-making in financial datasets.

## Project Overview

Financial fraud detection is a critical application of machine learning because fraudulent transactions are often rare, complex, and constantly evolving. In this project, multiple machine learning models are explored to build an effective fraud detection pipeline.

The development approach follows a progressive modeling strategy:
- **Linear Regression** was used as the baseline model to understand the dataset, workflow, and initial behavior of the features.
- **Random Forest** was then used as the primary model because of its ability to capture non-linear relationships and handle structured data effectively.
- **XGBoost** is planned as the next improvement step to further boost prediction performance and strengthen the risk analysis component.
- **SHAP** we plan on using shap to go above and beyond and add explainability as to how the transactions were flagged.

## Objectives

- Detect fraudulent financial transactions
- Analyze transaction risk levels
- Compare machine learning models for fraud detection
- Build a scalable and improvable fraud analysis pipeline

## Models Used

### 1. Linear Regression
Used as a baseline model to get an initial understanding of the data and establish a starting point for model comparison.

### 2. Random Forest
Used as the main model for prediction due to its robustness, better handling of feature interactions, and improved performance over the baseline approach.

### 3. XGBoost
Planned for future implementation to improve accuracy, efficiency, and predictive strength, especially for rich and complex dataset.

## Workflow

1. Data collection and preprocessing  
2. Feature engineering  
3. Train-test split  
4. Baseline modeling with Linear Regression  
5. Fraud prediction with Random Forest  
6. Model evaluation and performance comparison  
7. Future enhancement with XGBoost and SHAP

## Features

- Financial fraud transaction detection
- Baseline and advanced model comparison
- Extendable machine learning pipeline
- Scope for future optimization and tuning

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NumPy
- SMOTE
- RandomForest, Linear Regression, XGBoost (planned)
- SHapley Additive exPlanations(planned)

## Future Improvements

- Implement XGBoost for better performance
- Perform hyperparameter tuning
- Handle class imbalance more effectively
- Improve feature engineering
- Add visualization dashboard for fraud insights
- Deploy the model as a web application

## Conclusion

With this model our main target was to flag transactions with suspicious activties and optimise our model so that we reduce false alarms as much as we can. 
Furthermore adding explainability will help in learning as to why the transactions were flagged making it easier to catch fraud.
This project demonstrates a step-by-step machine learning approach to financial fraud detection and risk analysis. Starting with a baseline model and progressing toward more advanced ensemble methods makes the system both practical and scalable for future development.
