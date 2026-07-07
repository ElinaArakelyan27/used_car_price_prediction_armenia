import os
import warnings
import pandas as pd 
import numpy as np 
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "car_listings_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = datetime.now().year

# Load Dataset
df = pd.read_csv(DATA_PATH)
print("Rows before modelling:", len(df))

# Fixing issue
df["brand"] = df["brand"].replace("Land", "Land Rover")
df["engine_l"] = pd.to_numeric(df["engine_l"], errors="coerce")
df["engine_l"] = df["engine_l"].fillna(df["engine_l"].median())

# Removing outliers
df = df[df["price_usd"] <= 200000].copy()
print("Rows after removing outliers:", len(df))

# Group rare brands
brand_counts = df["brand"].value_counts()
rare_brands = brand_counts[brand_counts <= 2].index
df["brand"] = df["brand"].apply( lambda brand: "Other" if brand in rare_brands else brand)

# Feature Engineering
df["car_age"] = CURRENT_YEAR - df["year"]
df["km_per_year"] = df["mileage_km"] / df["car_age"].replace(0,1)
df["log_price"] = np.log(df["price_usd"])
print("New features created")
print(df[["car_age", "km_per_year", "log_price"]].head())


# Defining Features and target
NUMERIC_COLS =[
    "car_age",
    "mileage_km",
    "km_per_year",
    "engine_l"
]

CATEGORICAL_COLS = [
    "brand",
    "model", 
    "fuel",
    "transmission",
    "condition",
    "color"
]
for c in CATEGORICAL_COLS:
    df[c] = df[c].fillna("Unknown").astype(str)

X = df[
    NUMERIC_COLS + 
    CATEGORICAL_COLS
]
y = df["log_price"]
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size = 0.2,
    random_state = 42
)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERIC_COLS
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_COLS
        )
    ]
)

# Evaluation
def evaluate(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) 
    r2 = r2_score(y_true, y_pred)
    
    return mae, rmse, r2

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror", verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

param_dists = {
    "Ridge": {
        "model__alpha": uniform(0.1, 100)
    },

    "Random Forest": {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": randint(1, 5)
    },

    "XGBoost": {
        "model__n_estimators": randint(300, 800),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__subsample": uniform(0.7, 0.3)
    },

    "CatBoost": {
        "model__depth": randint(4, 10),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__iterations": randint(200, 800)
    }
} 

# Train and Tune
results = []
fitted_models = {}

for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    if name in param_dists:
        print(f"\nTuning {name}...")

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dists[name],
            n_iter=20,
            cv=5,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        cv_score = search.best_score_

    else:
        print(f"\nTraining {name}...")
        best_model = pipe.fit(X_train, y_train)
        cv_score = None

    preds = best_model.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, preds)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2": cv_score
    })

    fitted_models[name] = best_model

    print(f"{name} DONE")

    if cv_score is not None:
        print(f"Best CV R2: {cv_score:.4f}")

    print(f"Test R2: {r2:.4f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")

# Results Table

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R2", ascending=False)

print("\nModel Comparison")
print(results_df)

# Best model
best_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_models[best_name]

joblib.dump(
    {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "rare_brands": list(rare_brands),
        "current_year_used": CURRENT_YEAR,
        "median_mileage_by_age": (
            df.groupby("car_age")["mileage_km"]
              .median()
              .to_dict()
        ),
    },
    MODELS_DIR / "car_price_pipeline.joblib",
)
print(f"Saved best pipeline to {MODELS_DIR / 'car_price_pipeline.joblib'}")

# Feature Importance

rf_pipeline = fitted_models["Random Forest"]

feature_names = (
    NUMERIC_COLS +
    list(
        rf_pipeline.named_steps["prep"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLS)
    )
)

importances = rf_pipeline.named_steps["model"].feature_importances_

feat_imp = pd.Series(importances, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False).head(10)

plt.figure()
feat_imp.plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
plt.show()

# Model Comparison Plot
plt.figure()
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison.png", bbox_inches="tight")
plt.show()

def predict_price(
    brand, model_name, year, mileage_km=None,
    fuel="Unknown", transmission="Unknown", condition="Unknown", color="Unknown", engine_l = None,
    pipeline_path=MODELS_DIR / "car_price_pipeline.joblib",
):
  
    bundle = joblib.load(pipeline_path)
    pipe = bundle["pipeline"]
    rare = set(bundle["rare_brands"])
    cur_year = bundle["current_year_used"]
 
    brand_clean = "Other" if brand in rare else brand
    car_age = cur_year - year
    car_age_lookup = car_age if car_age > 0 else 1
 
    if mileage_km is None:
        mileage_km = bundle["median_mileage_by_age"].get(
            car_age_lookup, np.median(list(bundle["median_mileage_by_age"].values()))
        )
 
    km_per_year = mileage_km / (car_age if car_age != 0 else 1)
 
    row = pd.DataFrame([{ 
        "car_age": car_age,
        "mileage_km": mileage_km,
        "km_per_year": km_per_year,
        "brand": brand_clean,
        "model": model_name,
        "fuel": fuel,
        "transmission": transmission,
        "condition": condition,
        "color": color,
        "engine_l": engine_l
    }])
 
    log_pred = pipe.predict(row)[0]
    return float(np.exp(log_pred))
 
 

   