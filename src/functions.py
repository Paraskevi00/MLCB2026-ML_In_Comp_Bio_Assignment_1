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