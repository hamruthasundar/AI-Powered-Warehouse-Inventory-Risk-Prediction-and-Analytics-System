import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import accuracy_score

# LOAD DATASET
df = pd.read_csv("logistics_dataset.csv")

# CREATE TARGET
def detect_risk(row):

    if row['stock_level'] < row['reorder_point']:
        return "High Risk"

    elif row['stockout_count_last_month'] > 5:
        return "Medium Risk"

    else:
        return "Low Risk"

df['risk_status'] = df.apply(detect_risk, axis=1)

# FEATURES
X = df[
    [
        'stock_level',
        'reorder_point',
        'turnover_ratio',
        'KPI_score',
        'stockout_count_last_month'
    ]
]

# TARGET
y = df['risk_status']

# ENCODING
encoder = LabelEncoder()

y = encoder.fit_transform(y)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODELS
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "AdaBoost":
        AdaBoostClassifier()
}

best_accuracy = 0
best_model = None
best_model_name = ""

print("\nMODEL TRAINING RESULTS\n")

# TRAIN
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# SAVE MODEL
joblib.dump(best_model, "warehouse_model.pkl")

joblib.dump(encoder, "label_encoder.pkl")

print("\nBEST MODEL:", best_model_name)
print("BEST ACCURACY:", best_accuracy)

print("\nMODEL SAVED SUCCESSFULLY")