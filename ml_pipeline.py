import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

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
residuals = y_reg_test-linear_preds
print("\nLinear Regression Results")
print("RMSE:", linear_rmse)
print("R²:", linear_r2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

#Added plots for residuals and actual vs predicted
ax1.scatter(linear_preds,residuals)
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residual Plot for Linear Regression')

ax2.scatter(y_reg_test, linear_preds)
ax2.plot([y_reg_test.min(), y_reg_test.max()],
[y_reg_test.min(), y_reg_test.max()])
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title('Actual vs Predicted for Linear Regression')
plt.tight_layout
plt.show()

# ---| Random Forest Regressor |---
# Ensemble model that builds multiple decision trees and averages predictions.
# Captures nonlinear relationships and interactions between features.
def untuned_rfRegressor(X_train,y_reg_train,X_test,y_reg_test):
    rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
    )

    rf_reg.fit(X_train, y_reg_train)

    rf_reg_preds = rf_reg.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_reg_test, rf_reg_preds))
    rf_r2 = r2_score(y_reg_test, rf_reg_preds)

    print("\nUntuned Random Forest Regressor Results")
    print("RMSE:", rf_rmse)
    print("R²:", rf_r2)

    residuals_rf = y_reg_test - rf_reg_preds
    #Added plots for residuals and actual vs predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.scatter(rf_reg_preds,residuals_rf)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residual Plot (Untuned Random Forest)')

    ax2.scatter(y_reg_test, rf_reg_preds)
    ax2.plot(
    [y_reg_test.min(), y_reg_test.max()],
    [y_reg_test.min(), y_reg_test.max()]
    )

    ax2.set_xlabel('Actual Ratings')
    ax2.set_ylabel('Predicted Ratings')
    ax2.set_title('Actual vs Predicted (Untuned Random Forest)')
    plt.tight_layout()
    plt.show()
    #return rf_reg

untuned_rfRegressor(X_train,y_reg_train,X_test,y_reg_test) #Running the untuned RF Regressor

def tuned_rfRegressor(X_train,y_reg_train,X_test,y_reg_test,X,y_reg):
   
    param_grid = { #idk this has to go into the grid search tho
    "n_estimators": [100],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    }

    rf_reg = RandomForestRegressor(n_estimators=200,random_state=42)
    grid_rf_reg = GridSearchCV(estimator=rf_reg,param_grid=param_grid,
    cv=3,scoring="neg_mean_squared_error",n_jobs=2)
    grid_rf_reg.fit(X_train, y_reg_train)

    best_rf_reg = grid_rf_reg.best_estimator_

    cv_scores = cross_val_score(best_rf_reg,X,y_reg,cv=3, #adding cross-validation
    scoring="neg_mean_squared_error")

    cv_rmse = np.sqrt(-cv_scores)



    rf_reg_preds = best_rf_reg.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_reg_test, rf_reg_preds))
    rf_r2 = r2_score(y_reg_test, rf_reg_preds)
  
    print("\nTuned Random Forest Regressor Results")
    print("Best Params:", grid_rf_reg.best_params_)
    print("RMSE:", rf_rmse)
    print("R²:", rf_r2)
    print("\nCross-Validation RMSE Scores:", cv_rmse)
    print("Average CV RMSE:", cv_rmse.mean())
    residuals_rf = y_reg_test - rf_reg_preds
    #Added plots for residuals and actual vs predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.scatter(rf_reg_preds,residuals_rf)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residual Plot (Tuned Random Forest)')

    ax2.scatter(y_reg_test, rf_reg_preds)
    ax2.plot(
    [y_reg_test.min(), y_reg_test.max()],
    [y_reg_test.min(), y_reg_test.max()]
    )

    ax2.set_xlabel('Actual Ratings')
    ax2.set_ylabel('Predicted Ratings')
    ax2.set_title('Actual vs Predicted (Tuned Random Forest)')
    plt.tight_layout()
    plt.show()
    return best_rf_reg


tuned_rfRegressor(X_train,y_reg_train,X_test,y_reg_test,X,y_reg) #run the tuned model that also involveds cross-validation
best_rf_reg = tuned_rfRegressor(X_train,y_reg_train,X_test,y_reg_test,X,y_reg) #run the tuned model that also involveds cross-validation

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

#Adding plots for classification models: ROC and Precision vs. Recall Curves#

y_scores = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_clf_test, y_scores)
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(y_clf_test, y_scores)
ap_score = average_precision_score(y_clf_test, y_scores)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Logistic Regression)")
plt.legend()
plt.show()

plt.plot(recall, precision, label=f"AP = {ap_score:.2f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Logistic Regression)")
plt.legend()
plt.show()


# ---| Random Forest Classifier |---
# More powerful classifier that captures nonlinear decision boundaries and
# complex feature interactions.
def untuned_rfClassifier(X_train,y_clf_train,X_test,y_clf_test):
    rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
    )

    rf_clf.fit(X_train, y_clf_train)

    rf_clf_preds = rf_clf.predict(X_test)

    print("\nUntuned Random Forest Classifier Results")
    print("Accuracy:", accuracy_score(y_clf_test, rf_clf_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_clf_test, rf_clf_preds))
    print(classification_report(y_clf_test, rf_clf_preds))
    
    y_scores = rf_clf.predict_proba(X_test)[:, 1]
    # ---- ROC ----
    fpr, tpr, _ = roc_curve(y_clf_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # ---- Precision-Recall ----
    precision, recall, _ = precision_recall_curve(y_clf_test, y_scores)
    ap_score = average_precision_score(y_clf_test, y_scores)

    # ---- Plot ----
    plt.figure(figsize=(12, 5))

    # ROC subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Untuned Random Forest)")
    plt.legend()

    # PR subplot
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Untuned Random Forest)")
    plt.legend()

    plt.tight_layout()
    plt.show()

untuned_rfClassifier(X_train,y_clf_train,X_test,y_clf_test) #running the Untuned classification model

def tuned_rfClassifer(X_train,y_clf_train,y_clf_test,X_test,X,y_clf):
   
    param_grid = { #idk this has to go into the grid search tho
    "n_estimators": [100],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    }

    rf_clf = RandomForestClassifier(n_estimators=200,random_state=42)
    grid_rf_clf = GridSearchCV(estimator=rf_clf,param_grid=param_grid,
    cv=3,scoring="accuracy",n_jobs=2)
    grid_rf_clf.fit(X_train, y_clf_train)

    best_rf_clf = grid_rf_clf.best_estimator_

    cv_scores_clf = cross_val_score(best_rf_clf,X,y_clf, #adding cross validation
    cv=3,scoring="accuracy")

    rf_clf_preds = best_rf_clf.predict(X_test)
    print("\nTuned Random Forest Classifier Results")
    print("Best Params:", grid_rf_clf.best_params_)
    print("Accuracy:", accuracy_score(y_clf_test, rf_clf_preds))
    print(confusion_matrix(y_clf_test, rf_clf_preds))
    print(classification_report(y_clf_test, rf_clf_preds))
    print("\nCross-Validation Accuracy Scores:", cv_scores_clf)
    print("Average CV Accuracy:", cv_scores_clf.mean())
        # ---- ROC ----
    
    y_scores = best_rf_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_clf_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # ---- Precision-Recall ----
    precision, recall, _ = precision_recall_curve(y_clf_test, y_scores)
    ap_score = average_precision_score(y_clf_test, y_scores)

    # ---- Plot ----
    plt.figure(figsize=(12, 5))

    # ROC subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Tuned Random Forest)")
    plt.legend()

    # PR subplot
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Tuned Random Forest)")
    plt.legend()

    plt.tight_layout()
    plt.show()

tuned_rfClassifer(X_train,y_clf_train,y_clf_test,X_test,X,y_clf)  #running the tuned and cross validaiton model 

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

pca = PCA(n_components=2)
cluster_pca = pca.fit_transform(cluster_scaled)

# Add PCA components to dataframe
df["pca1"] = cluster_pca[:, 0]
df["pca2"] = cluster_pca[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df["pca1"],
    df["pca2"],
    c=df["cluster"],   # color by cluster
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters (PCA Visualization)")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

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
    best_rf_reg.feature_importances_,
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


cluster_summary = df.groupby("cluster")[[
    "averageRating",
    "log_votes",
    "runtimeMinutes",
    "movie_age"
]].mean()

print(cluster_summary)
# Save Dataset With Clusters
# Save updated dataset with cluster labels for further analysis
df.to_csv("imdb_movies_with_clusters.csv", index=False)

print("\nDone.")
print("Saved files:")
print("- elbow_plot.png")
print("- feature_importance.png")
print("- imdb_movies_with_clusters.csv")
