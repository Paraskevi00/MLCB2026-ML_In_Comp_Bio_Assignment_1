import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(data, test_size=0.2, seed=42):

    data = data.copy()

    # Create age bins for stratification
    data["age_bin"] = pd.cut(data["age"], bins=10)

    train, val = train_test_split(
        data,
        test_size=test_size,
        stratify=data["age_bin"],
        random_state=seed
    )

    train = train.drop(columns=["age_bin"])
    val = val.drop(columns=["age_bin"])

    return train, val




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_cpg_pipeline():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    return pipeline




from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor(cpg_cols, categorical_cols):

    cpg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("cpg", cpg_pipeline, cpg_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor



import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)

    return rmse, mae, r2, r



def bootstrap_evaluation(y_true, y_pred, n_bootstrap=1000, seed=42):

    np.random.seed(seed)

    metrics = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        y_t = y_true.iloc[idx]
        y_p = y_pred[idx]

        metrics.append(compute_metrics(y_t, y_p))

    metrics = np.array(metrics)

    results = {
        "rmse_mean": metrics[:,0].mean(),
        "rmse_ci": np.percentile(metrics[:,0], [2.5, 97.5]),

        "mae_mean": metrics[:,1].mean(),
        "mae_ci": np.percentile(metrics[:,1], [2.5, 97.5]),

        "r2_mean": metrics[:,2].mean(),
        "r2_ci": np.percentile(metrics[:,2], [2.5, 97.5]),

        "pearson_mean": metrics[:,3].mean(),
        "pearson_ci": np.percentile(metrics[:,3], [2.5, 97.5]),
    }

    return results, metrics