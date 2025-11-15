import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "lung_cancer_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_pipeline.pkl")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Features and target
target_column = "Lung Cancer Risk"
X = data.drop(columns=[target_column])
y = data[target_column]

# Detect numeric and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Using dataset: {DATA_PATH}")
print(f"Detected target column: {target_column}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Preprocessing
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # ✅ FIXED LINE

preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaler, numeric_features),
        ("cat", ohe, categorical_features)
    ]
)

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Save model
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model training complete. Saved to {MODEL_PATH}")
