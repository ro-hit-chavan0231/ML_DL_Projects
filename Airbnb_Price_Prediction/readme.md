## Project Description: Airbnb Price Prediction

This project aims to develop a robust prediction model for Airbnb listing prices using various machine learning and deep learning techniques. The objective is to assist both property owners in setting optimal prices and customers in evaluating fair prices, given minimal information about a property.

### 1. Data Loading and Initial Setup

The project begins by importing necessary libraries such as `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`, `sklearn`, and `tensorflow`. The `train.csv` dataset is loaded, and initial data type conversions are performed for columns like `id`, `accommodates`, `number_of_reviews` (to integer), `log_price`, `bathrooms`, `latitude`, `longitude`, `review_scores_rating`, `bedrooms`, `beds` (to float), and `cleaning_fee` (to boolean). The dataset is then split into training and testing sets to prepare for model development and evaluation.

### 2. Exploratory Data Analysis (EDA)

EDA was conducted to understand the data's characteristics and identify potential issues or insights:

*   **Data Overview:** `df_train.info()` and `df_train.describe()` provided summaries of data types and statistical distributions.
*   **Column Descriptions:** Detailed explanations for each column were provided to clarify their meaning.
*   **Price Distribution:** Histograms of `log_price` (raw and Z-score transformed) and a QQ plot were generated. The QQ plot indicated that `log_price` is not normally distributed, exhibiting heavier tails.
*   **Missing Data:** A heatmap visualization revealed significant missing values in columns such as `host_response_rate`, `first_review`, `last_review`, `review_scores_rating`, and `thumbnail_url`. This highlighted areas requiring imputation or careful handling.
*   **Feature-Specific Analysis:**
    *   **`first_review` and `last_review`:** Plots analyzed the count of reviews by year and their relationship with `log_price`. It was observed that many listings lacked these review dates, and while median prices were consistent, price variability was higher for listings with missing review information. The number of listings and reviews showed significant growth over the years, especially between 2013-2017.
    *   **`host_since`:** Analysis of host join dates indicated a steady growth in new hosts, particularly between 2013 and 2016. `log_price` appeared stable regardless of when a host joined, but with a wide range of prices within each year.
    *   **Categorical Features:** `value_counts()` was used to explore the distribution of `property_type`, `room_type`, `bed_type`, `cancellation_policy`, `city`, `host_response_rate`, `review_scores_rating`, `bathrooms`, and `host_has_profile_pic`. This helped identify popular categories and potential data cleaning needs.
*   **Geographical Distribution:** A map visualization (`create_map`) was used to display Airbnb prices across different cities (e.g., NYC), showing the spatial distribution of `log_price`.

### 3. Data Preparation and Outlier Handling
**Dataset**
 The dataset which is used here, is collected from Kaggle website. Here is the link of the dataset : https://www.kaggle.com/stevezhenghp/airbnb-price-prediction.

Several steps were taken to prepare the data for modeling:

*   **Outlier Detection:** The distribution of `log_price` was analyzed to identify extreme values. Listings with `log_price` less than 2.5 or greater than 7.5 were flagged, and listings with `log_price` equal to 0 were dropped as outliers, indicating incorrect or placeholder values.
*   **Amenity Extraction:** The `amenities` column, which contained text descriptions, was processed to extract unique amenities and convert them into a set of dummy variables.
*   **Feature Engineering and Dummification:** A `processing` function was defined to handle various transformations:
    *   Categorical columns (`property_type`, `cancellation_policy`, `first_review`, `neighbourhood`) were processed to group less frequent categories into 'other' or fill missing values.
    *   `host_response_rate` was converted to a numerical format.
    *   Missing numerical values (`review_scores_rating`, `bathrooms`, `bedrooms`, `beds`) were filled with default values (e.g., 0 or -1).
    *   Boolean columns (`cleaning_fee`, `host_has_profile_pic`, `host_identity_verified`, `instant_bookable`) were converted to 0/1.
    *   `host_since` was converted to the year the host joined, with missing values filled.
    *   All relevant categorical features were converted into dummy variables using `pd.get_dummies`.
    *   Irrelevant columns (`amenities`, `thumbnail_url`, `description`, `id`, `last_review`, `zipcode`, `name`) were dropped.
*   **Column Standardization:** A `columns_standardization` function ensured that the training and testing datasets had consistent columns after the feature engineering process.

### 4. Prediction Model Deployment

Five different regression algorithms were chosen for price prediction:

*   **Linear Regression:** A basic linear model to establish a baseline performance.
*   **Random Forest Regressor:** An ensemble tree-based method known for its robustness and accuracy.
*   **Gradient Boosting Regressor:** Another powerful ensemble technique that builds models sequentially.
*   **Dense Neural Networks:** A deep learning approach using sequential layers.

For each model:
*   The model was trained on the processed training data (`df_d`).
*   A `show_metrics` function displayed key evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), RMSE ratio for test and train, and R-squared for both test and train sets.
*   An `analysis` function generated plots showing the correlation between predicted and real values, and the distribution of prediction errors, to visually assess model performance.
*   For the Dense Neural Network, `MinMaxScaler` was applied to scale the features, and `EarlyStopping` was used during training to prevent overfitting. The training history (loss and validation loss) was also plotted.

This structured approach allowed for comprehensive data understanding, preparation, and the evaluation of multiple predictive models to achieve the project's goal of accurate Airbnb price prediction.


