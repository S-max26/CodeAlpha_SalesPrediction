# advertising_sales_regression_v2.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the advertising dataset
data_path = "C:/Users/shash/OneDrive/Desktop/Internship 2/task 4/Advertising.csv"
ad_data = pd.read_csv(data_path)

# Quick data check
print("📊 Preview of Dataset:\n", ad_data.head())
print("\nℹ️ Dataset Info:\n")
ad_data.info()
print("\n🧹 Missing Values:\n", ad_data.isnull().sum())

# Visualize feature correlations
plt.figure(figsize=(8, 6))
sns.heatmap(ad_data.corr(), annot=True, cmap="viridis", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Define features and target variable
features = ad_data[['TV', 'Radio', 'Newspaper']]
target = ad_data['Sales']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Initialize and train linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict sales on test set
sales_predictions = reg_model.predict(X_test)

# Evaluate the model
mse_score = mean_squared_error(y_test, sales_predictions)
r2 = r2_score(y_test, sales_predictions)

print(f"\n📉 Mean Squared Error: {mse_score:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Plot Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=sales_predictions, color='tomato')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
