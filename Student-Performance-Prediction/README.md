# Student Performance Prediction

This notebook analyzes student performance data using various regression models.

## Data Description

The dataset contains information about students' performance in Portuguese language courses. It includes demographic, social, and school-related features, as well as final grades (G1, G2, G3).

## Analysis Steps

1.  **Data Exploration:** Initial inspection of the dataset, including shape, column names, data types, and summary statistics. Checked for missing values.
2.  **Data Visualization:** Visualizations to understand the distribution of key features such as school, sex, age, family size, parent's cohabitation status, parental education, parental jobs, reason for choosing the school, guardian, travel time, study time, failures, extra educational support, family support, paid classes, activities, nursery attendance, desire for higher education, internet access, romantic relationships, family relationship quality, free time, going out with friends, workday and weekend alcohol consumption, health status, and absences. Also visualized the distribution of G1 and G2 grades.
3.  **Data Training:** Preparation of the data for model training by selecting relevant features and splitting the dataset into training and testing sets.
4.  **Model Creation:** Implementation and training of four regression models:
    *   Linear Regression
    *   Lasso Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
5.  **Performance Evaluation:** Evaluation of each model's performance using Root Mean Square Error (RMSE) and R-squared (accuracy) on the training data. The Decision Tree Regressor showed the highest accuracy on the training data, indicating potential overfitting, while Linear Regression and Lasso Regression showed comparable performance.
