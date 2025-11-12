❤️ Heart Disease Prediction Using Machine Learning
📘 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction of heart disease can help in timely treatment and lifestyle changes.
This project aims to predict the likelihood of a person having heart disease using multiple Machine Learning algorithms and compare their performance to find the best model.

The system takes patient health parameters (like age, blood pressure, cholesterol, etc.) and predicts whether they are likely to have heart disease (1) or not (0).

🎯 Objectives

Build a robust predictive model for heart disease diagnosis.

Compare performance of different Machine Learning algorithms.

Deploy the model using Flask for an interactive web application.

🧠 Machine Learning Models Used

The following algorithms were implemented and evaluated:

Model	Description
K-Nearest Neighbors (KNN)	A simple distance-based algorithm for classification.
Logistic Regression	A linear model commonly used for binary classification.
Naive Bayes	A probabilistic model based on Bayes' theorem.
Decision Tree Classifier	A tree-based model that splits data based on feature importance.
Random Forest Classifier	An ensemble model using multiple decision trees for better accuracy.
AdaBoost Classifier	A boosting algorithm that combines weak learners into a strong model.
Gradient Boosting Classifier	An advanced ensemble model that optimizes prediction errors iteratively.
XGBoost Classifier	An optimized gradient boosting model offering high accuracy and speed.
Support Vector Machine (SVC)	A margin-based classifier that performs well on high-dimensional data.
📊 Dataset Description

The dataset used is the Heart Disease Dataset, typically available from the UCI Machine Learning Repository
 or Kaggle.

Feature	Description
age	Age of the person
sex	Gender (1 = male, 0 = female)
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar > 120 mg/dl
restecg	Resting electrocardiographic results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina
oldpeak	ST depression induced by exercise
slope	Slope of the peak exercise ST segment
ca	Number of major vessels colored by fluoroscopy
thal	Thalassemia
target	1 = heart disease, 0 = no heart disease
