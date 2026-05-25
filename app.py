# ============================================================
# Project Name : Product Demand Prediction
# Tool         : PyCharm
# Purpose      : To predict future product demand using ML models
# ============================================================


# ---------------- IMPORT REQUIRED LIBRARIES ----------------

# pandas is used for reading CSV files and handling tabular data
import pandas as pd

# numpy is used for numerical calculations
import numpy as np

# Flask is used to build a web-based prediction application
from flask import Flask, render_template, request

# matplotlib is used for data visualization (graphs & charts)
import matplotlib.pyplot as plt

# LabelEncoder converts categorical text data into numeric form
from sklearn.preprocessing import LabelEncoder

# Used to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Used to evaluate regression model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# XGBoost – advanced boosting regression algorithm
from xgboost import XGBRegressor

# KMeans clustering for demand categorization
from sklearn.cluster import KMeans

# Accuracy metric (used for KMeans classification comparison)
from sklearn.metrics import accuracy_score

# mode is used to find most frequent demand label in clusters
from scipy.stats import mode


# ---------------- FLASK APPLICATION ----------------

# Create Flask app object
app = Flask(__name__)


# ---------------- STEP 1: LOAD DATASET ----------------

# Read product sales dataset from CSV file
data = pd.read_csv("ProductData.csv")

# Display dataset to understand raw input
print("\nOriginal Dataset:\n", data)


# ---------------- STEP 2: DATA PREPROCESSING ----------------

# Check for missing (NULL) values in each column
print("\nMissing Values:\n", data.isnull().sum())

# Numeric columns where mean value is suitable
numeric_cols = ['product_id', 'price', 'holiday']

# Categorical columns where most frequent value is suitable
categorical_cols = ['month', 'product']

# Fill missing numeric values with column mean
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())
    # fillna(mean) prevents errors during model training

# Fill missing categorical values with most common value (mode)
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Remove rows where output variable (units_sold) is missing
# Model cannot learn without target value
data.dropna(subset=['units_sold'], inplace=True)

# Remove duplicate rows to avoid biased learning
data.drop_duplicates(inplace=True)


# ---------------- STEP 3: FEATURE ENGINEERING ----------------

# Convert month column into datetime format
data['month'] = pd.to_datetime(data['month'])

# Extract year from month
data['year'] = data['month'].dt.year

# Extract numeric month value (1–12)
data['month_num'] = data['month'].dt.month

# Encode categorical columns into numeric values
# ML models only understand numbers, not text
for col in ['product_id', 'store_id', 'product']:
    data[col] = LabelEncoder().fit_transform(data[col])


# ---------------- STEP 4: SELECT INPUT & OUTPUT ----------------

# Input features used for prediction
X = data[['year', 'month_num', 'product_id', 'store_id', 'price', 'holiday']]

# Output variable (target)
y = data['units_sold']


# ---------------- STEP 5: TRAIN–TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% data used for testing
    random_state=42      # Ensures same split every run
)
#Training data is used to learn patterns, and testing data is used to evaluate performance.”

# ---------------- STEP 6: LINEAR REGRESSION MODEL ----------------

# Create Linear Regression model
lr = LinearRegression()

# Train model using training data
lr.fit(X_train, y_train)

# Predict demand for test data
y_pred_lr = lr.predict(X_test)

# Calculate performance metrics
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_acc = r2_score(y_test, y_pred_lr)


# ---------------- STEP 7: RANDOM FOREST REGRESSION ----------------

# Create Random Forest model with 200 trees
rf = RandomForestRegressor(n_estimators=200, random_state=42)

# Train Random Forest model
rf.fit(X_train, y_train)

# Predict demand
y_pred_rf = rf.predict(X_test)

# Performance metrics
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_acc = r2_score(y_test, y_pred_rf)


# ---------------- STEP 8: DECISION TREE REGRESSION ----------------

# Create Decision Tree with depth control to avoid overfitting
dt = DecisionTreeRegressor(
    max_depth=8,
    min_samples_split=10,
    random_state=42
)

# Train Decision Tree model
dt.fit(X_train, y_train)

# Predict demand
y_pred_dt = dt.predict(X_test)

# Performance metrics
dt_mae = mean_absolute_error(y_test, y_pred_dt)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
dt_acc = r2_score(y_test, y_pred_dt)


# ---------------- STEP 9: KNN REGRESSION ----------------

# Create KNN model with 5 neighbors
knn = KNeighborsRegressor(n_neighbors=5)

# Train KNN model
knn.fit(X_train, y_train)

# Predict demand
y_pred_knn = knn.predict(X_test)

# Performance metrics
knn_mae = mean_absolute_error(y_test, y_pred_knn)
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_acc = r2_score(y_test, y_pred_knn)


# ---------------- STEP 10: SUPPORT VECTOR REGRESSION ----------------

# Create SVR model using RBF kernel
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Train SVR model
svr.fit(X_train, y_train)

# Predict demand
y_pred_svr = svr.predict(X_test)

# Performance metrics
svr_mae = mean_absolute_error(y_test, y_pred_svr)
svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
svr_acc = r2_score(y_test, y_pred_svr)


# ---------------- STEP 11: XGBOOST REGRESSION ----------------

# Create XGBoost model with tuned parameters
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train XGBoost model
xgb.fit(X_train, y_train)

# Predict demand
y_pred_xgb = xgb.predict(X_test)

# Performance metrics
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_acc = r2_score(y_test, y_pred_xgb)


# ---------------- STEP 12: KMEANS CLUSTERING ----------------

# Create demand categories:
# 0 = Low Demand, 1 = Medium Demand, 2 = High Demand
data['demand_label'] = pd.qcut(
    data['units_sold'],
    q=3,
    labels=[0, 1, 2]
)

# Convert demand labels to integer
y_demand = data['demand_label'].astype(int)

# Create KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit clustering model
kmeans.fit(X)

# Get cluster labels
kmeans_labels = kmeans.labels_

# Map clusters to actual demand labels
kmeans_pred = np.zeros_like(kmeans_labels)

for i in np.unique(kmeans_labels):
    mask = (kmeans_labels == i)
    kmeans_pred[mask] = mode(y_demand[mask])[0]

# Calculate clustering accuracy
kmeans_acc = accuracy_score(y_demand, kmeans_pred)


# ---------------- STEP 13: MODEL COMPARISON ----------------

print("\n------- MODEL COMPARISON -------")
print(f"Linear Regression | MAE:{lr_mae:.2f} | RMSE:{lr_rmse:.2f} | Acc:{lr_acc*100:.2f}%")
print(f"Random Forest     | MAE:{rf_mae:.2f} | RMSE:{rf_rmse:.2f} | Acc:{rf_acc*100:.2f}%")
print(f"Decision Tree     | MAE:{dt_mae:.2f} | RMSE:{dt_rmse:.2f} | Acc:{dt_acc*100:.2f}%")
print(f"KNN               | MAE:{knn_mae:.2f} | RMSE:{knn_rmse:.2f} | Acc:{knn_acc*100:.2f}%")
print(f"SVR               | MAE:{svr_mae:.2f} | RMSE:{svr_rmse:.2f} | Acc:{svr_acc*100:.2f}%")
print(f"XGBoost           | MAE:{xgb_mae:.2f} | RMSE:{xgb_rmse:.2f} | Acc:{xgb_acc*100:.2f}%")
print("\nKMeans Accuracy:", round(kmeans_acc * 100, 2), "%")


# ---------------- STEP 14: HYBRID MODEL ----------------

# Hybrid prediction using average of RF and XGB
hybrid_pred = (y_pred_rf + y_pred_xgb) / 2

# Hybrid performance
hybrid_mae = mean_absolute_error(y_test, hybrid_pred)
hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
hybrid_acc = r2_score(y_test, hybrid_pred)

print("\n------- HYBRID MODEL (RF + XGB) -------")
print("MAE:", round(hybrid_mae, 2))
print("RMSE:", round(hybrid_rmse, 2))
print("Accuracy:", round(hybrid_acc * 100, 2), "%")


# ---------------- STEP 15: VISUALIZATION ----------------

models = ['LR', 'RF', 'DT', 'KNN', 'SVR', 'XGB', 'KMeans', 'Hybrid']
acc_values = [lr_acc, rf_acc, dt_acc, knn_acc, svr_acc, xgb_acc, kmeans_acc, hybrid_acc]

plt.figure()
plt.bar(models, acc_values)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison using Accuracy")
plt.show()


# ---------------- STEP 16: FLASK ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    year = int(request.form["year"])
    month_num = int(request.form["month_num"])
    product_id = int(request.form["product_id"])
    store_id = int(request.form["store_id"])
    price = float(request.form["price"])
    holiday = int(request.form["holiday"])

    future_data = pd.DataFrame({
        "year": [year],
        "month_num": [month_num],
        "product_id": [product_id],
        "store_id": [store_id],
        "price": [price],
        "holiday": [holiday]
    })

    # Predict future demand using XGBoost
    prediction = xgb.predict(future_data)[0]

    return render_template("index.html", prediction=int(prediction))


# ---------------- RUN FLASK APP ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
