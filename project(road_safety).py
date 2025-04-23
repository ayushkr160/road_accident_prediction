# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 02:16:38 2025

@author: AYUSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# Load the dataset
df = pd.read_csv("D:\\INT 557 (Data Science with python)\\road_safety.csv", encoding='cp1252')
df.info()
df.describe()
#Accident Severity Distribution (Pie chart)
severity_counts = df['Casualty Severity'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Accident Severity Distribution')
plt.axis('equal')
plt.show()

#Accidents by Road Surface(Bar Plot)
surface_counts = df['Road Surface'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=surface_counts.index, y=surface_counts.values)
plt.title('Accidents by Road Surface')
plt.xlabel('Road Surface')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

#Accidents by Lighting Conditions (Horizontal Bar Plot)
lighting_counts = df['Lighting Conditions'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=lighting_counts.values, y=lighting_counts.index, palette='magma')
plt.title('Accidents by Lighting Conditions')
plt.xlabel('Number of Accidents')
plt.ylabel('Lighting Conditions')
plt.show()

#Accidents by Weather conditions(Donut Chart)
weather_counts = df['Weather Conditions'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
plt.title('Accidents by Weather Conditions')
plt.axis('equal')
plt.show()

#Hourly distribution of Accidents(Line Plot)
hourly_counts = df.groupby('Time (24hr)').size()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o')
plt.title('Hourly Distribution of Accidents')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(range(0,24))
plt.grid(True)
plt.show()

#Accidents by type of Vehicle (Horizontal Bar Plot)
vehicle_counts = df['Type of Vehicle'].value_counts().head(10)  # Top 10 vehicles

plt.figure(figsize=(10, 6))
sns.barplot(x=vehicle_counts.values, y=vehicle_counts.index, palette='coolwarm')
plt.title('Top 10 Vehicle Types Involved in Accidents')
plt.xlabel('Number of Accidents')
plt.ylabel('Vehicle Type')
plt.show()

#Age distribution of Casualties(Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['Age of Casualty'], bins=20, kde=True)
plt.title('Age Distribution of Casualties')
plt.xlabel('Age of Casualty')
plt.ylabel('Number of Casualties')
plt.grid(True)
plt.show()

#Accidents by Casualty Class and Severity(Grouped Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(x='Casualty Class', hue='Casualty Severity', data=df)
plt.title('Accidents by Casualty Class and Severity')
plt.xlabel('Casualty Class')
plt.ylabel('Number of Accidents')
plt.legend(title='Casualty Severity')
plt.show()


df = pd.read_csv("D:\\INT 557 (Data Science with python)\\road_safety.csv", encoding='cp1252')

# Extract hour from time
df['Hour'] = df['Time (24hr)'] // 100

# Define features for modeling
features = ['Road Surface', 'Lighting Conditions', 'Weather Conditions',
            'Hour', 'Type of Vehicle', 'Number of Vehicles',
            'Casualty Class', 'Age of Casualty', 'Sex of Casualty']

# Label Encoding for categorical features
le = LabelEncoder()
for col in features:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Encode target variable for classification: Casualty Severity
df['Casualty Severity'] = le.fit_transform(df['Casualty Severity'])

# Prepare data
X = df[features]
y_classification = df['Casualty Severity']
y_regression = df['Age of Casualty']

# Split data for classification and regression
X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X, y_classification, y_regression, test_size=0.3, random_state=42)

# ------------------------------------------
# 1. Logistic Regression (Classification)
# ------------------------------------------
print("\n--- Logistic Regression ---")
log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train, y_train_cls)
log_pred = log_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test_cls, log_pred))
print(classification_report(y_test_cls, log_pred))

# 2. Support Vector Machine (Classification)
# ------------------------------------------
print("\n--- Support Vector Machine (SVM) ---")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train_cls)
svm_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test_cls, svm_pred))
print(classification_report(y_test_cls, svm_pred))

# ------------------------------------------
# 3. K-Nearest Neighbors (Classification)
# ------------------------------------------
print("\n--- K-Nearest Neighbors (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train_cls)
knn_pred = knn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test_cls, knn_pred))
print(classification_report(y_test_cls, knn_pred))


# 4. Linear Regression (Predicting Age)
# ------------------------------------------
print("\n--- Linear Regression (Predicting Age of Casualty) ---")
lin_model = LinearRegression()
lin_model.fit(X_train, y_train_reg)
lin_pred = lin_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, lin_pred))
print("RMSE:",rmse)
