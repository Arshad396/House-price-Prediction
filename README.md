# House-price-Prediction
House Price Prediction using Random Forest Regressor
This project aims to predict house prices based on a dataset containing various features of houses. The model used in this project is a Random Forest Regressor, a popular machine learning algorithm for regression tasks due to its robustness and accuracy. The project walks through the data preprocessing steps, model training, evaluation, and feature importance analysis.

Table of Contents
Project Overview
Dataset
Preprocessing
Model Training
Evaluation
Feature Importance
Conclusion
Project Overview
This project utilizes machine learning to predict house prices. The goal is to build a model that can generalize well to unseen data and provide accurate predictions. We use a Random Forest Regressor for this task because of its ability to handle non-linear relationships and its resistance to overfitting.

Dataset
The dataset consists of features such as:

Lot size
Number of rooms
Location
Year built
Property type
These features are used to predict the target variable, house price.

Preprocessing
The data is first preprocessed to handle missing values, standardize numerical features, and encode categorical variables. The preprocessing steps include:

Handling missing data by either imputing or dropping rows.
Standardizing the numerical features for better performance of the model.
Encoding categorical features using one-hot encoding to make them suitable for the Random Forest algorithm.
Model Training
The Random Forest Regressor was chosen for its ensemble learning capability and tendency to provide accurate results without extensive hyperparameter tuning. The training steps included:

Splitting the dataset into training and test sets.
Normalizing the data to improve model convergence.
Training the Random Forest Regressor on the training data.
Evaluation
The model was evaluated using the Mean Squared Error (MSE), which measures the average of the squared differences between actual and predicted values. Cross-validation was also performed to assess the model’s generalization ability. Below is the evaluation code:

python
Copy code
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-validated MSE:", -scores.mean())
Feature Importance
One of the strengths of the Random Forest algorithm is its ability to provide insights into feature importance. The features that had the most influence on the house price predictions were:

Location
Number of rooms
Lot size
A bar chart showing the feature importances was generated:

python
Copy code
import matplotlib.pyplot as plt

feature_importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
Conclusion
The Random Forest Regressor performed well in predicting house prices, with a satisfactory Mean Squared Error. The model was also able to provide valuable insights into the most important features that affect house prices, such as location, lot size, and number of rooms.
