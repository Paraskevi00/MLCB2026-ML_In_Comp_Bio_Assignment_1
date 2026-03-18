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