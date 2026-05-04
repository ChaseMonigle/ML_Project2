import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Starting IMDb Movie Rating ML Pipeline...")


# Load Data:
# Load the preprocessed IMDb dataset. This dataset has already been cleaned and contains
# only relevant movie records with engineered features such as log-transformed votes and 
# movie age.
df = pd.read_csv("imdb_movies_ml_ready.csv")

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# Set Features and Targets:
# Define target variables for both regression and classification tasks. 
# 1. averageRating is used for regression (continuous prediction)
# 2. is_good is a binary label (1 if rating >= 7, else 0)
y_reg = df["averageRating"]
y_clf = df["is_good"]

# CREATE feature matrix via removing target columns
X = df.drop(columns=["averageRating", "is_good"])

# THEN clean X:
# Remove columns that are irrelevant or may introduce noise into the model. These include 
# identifiers or episode-related metadata not useful for prediction.
X = X.drop(columns=["parentTconst", "seasonNumber", "episodeNumber"], errors="ignore")
X = X.drop(columns=["endYear"], errors="ignore")

# THEN keep only numeric features 
X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

print("Features being used:", X.columns.tolist())


# Train/Test Split
# Split dataset(80/20) training and testing sets.
# random_state ensures reproducibility of results.
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X,
    y_reg,
    y_clf,
    test_size=0.2,
    random_state=42
)


# Scale Features
# Standardize features to have mean = 0 and variance = 1.
# This is important for models like Linear Regression and Logistic Regression
# which are sensitive to feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Regression Models

print("\nRunning Regression Models...")

# ---| Linear Regression (Baseline Model)|--- 
# Used to determine whether a simple linear relationship exists
# between input features and movie ratings.
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_reg_train)

linear_preds = linear_model.predict(X_test_scaled)

# Eavluate using RMSE and R^2
linear_rmse = np.sqrt(mean_squared_error(y_reg_test, linear_preds))
linear_r2 = r2_score(y_reg_test, linear_preds)

print("\nLinear Regression Results")
print("RMSE:", linear_rmse)
print("R²:", linear_r2)

# ---| Random Forest Regressor |---
# Ensemble model that builds multiple decision trees and averages predictions.
# Captures nonlinear relationships and interactions between features.
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_reg.fit(X_train, y_reg_train)

rf_reg_preds = rf_reg.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_reg_test, rf_reg_preds))
rf_r2 = r2_score(y_reg_test, rf_reg_preds)

print("\nRandom Forest Regressor Results")
print("RMSE:", rf_rmse)
print("R²:", rf_r2)


# Classification Models

print("\nRunning Classification Models...")

# ---| Logistic Regression (Baseline Classifier) |---
# Used to classify whether a movie is "good" based on a linear decision boundary.
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_clf_train)

log_preds = log_model.predict(X_test_scaled)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_clf_test, log_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test, log_preds))
print(classification_report(y_clf_test, log_preds))

# ---| Random Forest Classifier |---
# More powerful classifier that captures nonlinear decision boundaries and
# complex feature interactions.
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_clf.fit(X_train, y_clf_train)

rf_clf_preds = rf_clf.predict(X_test)

print("\nRandom Forest Classifier Results")
print("Accuracy:", accuracy_score(y_clf_test, rf_clf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test, rf_clf_preds))
print(classification_report(y_clf_test, rf_clf_preds))

# K-Means Clustering

print("\nRunning K-Means Clustering...")

# Select key features for Clustering reps movie performance and characteristics
cluster_features = df[
    [
        "averageRating",
        "log_votes",
        "runtimeMinutes",
        "movie_age"
    ]
]

# Scale clustering features to ensure equal weighting
cluster_scaled = scaler.fit_transform(cluster_features)

# Apply K-Means clustering to group similar movies 
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(cluster_scaled)

print("\nCluster Counts:")
print(df["cluster"].value_counts())


# Elbow Method Plot

print("\nCreating elbow plot...")

# Determine optimal number of clusters using inertia
inertia = []

for k in range(1, 10):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(cluster_scaled)
    inertia.append(km.inertia_)

# Plot inertia vs number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means Clustering")
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()
plt.close()

# Feature Importance Plot

print("\nCreating feature importance plot...")

# Extract feature importance scores from Random Forest Regressor
feature_importance = pd.Series(
    rf_reg.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_features = feature_importance.head(10)

print("\nTop 10 Features Affecting Movie Ratings:")
print(top_features)

# Visualize Top features 
plt.figure(figsize=(10, 6))
top_features.plot(kind="bar")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Top Features Affecting Movie Ratings")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
plt.close()

# Save Dataset With Clusters
# Save updated dataset with cluster labels for further analysis
df.to_csv("imdb_movies_with_clusters.csv", index=False)

print("\nDone.")
print("Saved files:")
print("- elbow_plot.png")
print("- feature_importance.png")
print("- imdb_movies_with_clusters.csv")
