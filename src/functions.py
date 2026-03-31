import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

def dataset_summary(X, y, name):
    print(f"\n{name} SET")
    print(f"Samples: {len(X)}")
    print(f"Age mean: {y.mean():.2f}")
    print(f"Age std: {y.std():.2f}")
    print(f"Age min: {y.min()}, max: {y.max()}")
    print("Sex distribution:")
    print(X["sex"].value_counts(normalize=True))


def evaluate_bootstrap(y_true, y_pred, n_iterations=1000, seed=42):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    np.random.seed(seed)
    n = len(y_true)

    rmse_list, mae_list, r2_list, pearson_list = [], [], [], []

    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)

        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]

        rmse_list.append(np.sqrt(mean_squared_error(y_true_bs, y_pred_bs)))
        mae_list.append(mean_absolute_error(y_true_bs, y_pred_bs))
        r2_list.append(r2_score(y_true_bs, y_pred_bs))
        pearson_list.append(pearsonr(y_true_bs, y_pred_bs)[0])

    def ci(arr):
        return np.percentile(arr, [2.5, 97.5])

    results = {
        "rmse": rmse_list,
        "mae": mae_list,
        "r2": r2_list,
        "pearson": pearson_list,
        "summary": {
            "RMSE": (np.mean(rmse_list), ci(rmse_list)),
            "MAE": (np.mean(mae_list), ci(mae_list)),
            "R2": (np.mean(r2_list), ci(r2_list)),
            "Pearson r": (np.mean(pearson_list), ci(pearson_list)),
        }
    }

    return results


def print_results(name, results):
    print(f"\n{name}")
    print("-" * 40)

    for metric, (mean, ci) in results["summary"].items():
        print(f"{metric}: {mean:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")


def format_results(model, stage, results):
    return {
        "Model": model,
        "Stage": stage,
        "RMSE (mean)": f"{np.mean(results['rmse']):.2f}",
        "95% CI": f"[{np.percentile(results['rmse'],2.5):.2f}, {np.percentile(results['rmse'],97.5):.2f}]",
        "MAE": f"{np.mean(results['mae']):.2f}",
        "R2": f"{np.mean(results['r2']):.2f}",
        "Pearson r": f"{np.mean(results['pearson']):.2f}"
    }


def summarize_results(name, results):
    return {
        "Model": name,
        "RMSE": f"{np.mean(results['rmse']):.2f} [{np.percentile(results['rmse'],2.5):.2f}, {np.percentile(results['rmse'],97.5):.2f}]",
        "MAE": f"{np.mean(results['mae']):.2f}",
        "R2": f"{np.mean(results['r2']):.2f}",
        "Pearson r": f"{np.mean(results['pearson']):.2f}"
    }

def prepare_boxplot_data(results, model_name):
    return pd.DataFrame({
        "RMSE": results["rmse"],
        "R2": results["r2"],
        "Model": model_name
    })


from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def mrmr_feature_selection(X, y, K=100):
    import numpy as np
    from scipy.stats import pearsonr

    X_np = X.to_numpy()
    cols = list(X.columns)

    # relevance
    relevance = []
    for i in range(X_np.shape[1]):
        r, _ = pearsonr(X_np[:, i], y)
        relevance.append(abs(r))
    relevance = np.array(relevance)

    # redundancy
    corr = np.abs(np.corrcoef(X_np, rowvar=False))

    selected = [np.argmax(relevance)]
    remaining = list(range(len(cols)))
    remaining.remove(selected[0])

    for _ in range(K - 1):
        scores = []
        for j in remaining:
            rel = relevance[j]
            red = np.mean(corr[j, selected])
            scores.append(rel - red)

        best_idx = remaining[np.argmax(scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [cols[i] for i in selected]



from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_features]
    


class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)
    


from sklearn.pipeline import Pipeline

def build_final_pipeline(preprocessor, selected_features, best_model):
    feature_names = preprocessor.get_feature_names_out()
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("to_df", ToDataFrame(feature_names)),
        ("feature_selection", FeatureSelector(selected_features)),
        ("model", best_model)
    ])
    
    return pipeline



def plot_predictions(y_true, y_pred, title, filename):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(title)

    # diagonal line
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color="red")

    plt.savefig(filename)
    plt.show()



import optuna
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR


def optuna_tune_model(model_name, pipeline, X_train, y_train, n_trials=40, cv=5):
    
    def objective(trial):
        
        if model_name == "ElasticNet":
            alpha = trial.suggest_float("alpha", 0.001, 10, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        elif model_name == "SVR":
            C = trial.suggest_float("C", 0.1, 500, log=True)
            epsilon = trial.suggest_categorical("epsilon", [0.01, 0.1, 0.5, 1.0])
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])

            model = SVR(C=C, epsilon=epsilon, kernel=kernel)

        elif model_name == "BayesianRidge":
            alpha_1 = trial.suggest_float("alpha_1", 1e-7, 1e-3, log=True)
            alpha_2 = trial.suggest_float("alpha_2", 1e-7, 1e-3, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True)

            model = BayesianRidge(
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2
            )

        else:
            raise ValueError("Unknown model")

        # update pipeline model
        pipeline.set_params(model=model)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error"
        )

        return -np.mean(scores)  # RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # best model
    best_params = study.best_params

    print("Best params:", best_params)
    print("Best RMSE:", study.best_value)

    # rebuild best model
    if model_name == "ElasticNet":
        best_model = ElasticNet(**best_params)
    elif model_name == "SVR":
        best_model = SVR(**best_params)
    elif model_name == "BayesianRidge":
        best_model = BayesianRidge(**best_params)

    pipeline.set_params(model=best_model)

    pipeline.fit(X_train, y_train)

    return pipeline, study



from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

def evaluate_bootstrap_classification(y_true, y_pred, y_prob, n_iterations=1000, seed=42):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    np.random.seed(seed)
    n = len(y_true)

    acc_list, f1_list, mcc_list, roc_list, pr_list = [], [], [], [], []

    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)

        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        y_prob_bs = y_prob[indices]

        acc_list.append(accuracy_score(y_true_bs, y_pred_bs))
        f1_list.append(f1_score(y_true_bs, y_pred_bs))
        mcc_list.append(matthews_corrcoef(y_true_bs, y_pred_bs))

        try:
            roc_list.append(roc_auc_score(y_true_bs, y_prob_bs))
            pr_list.append(average_precision_score(y_true_bs, y_prob_bs))
        except:
            roc_list.append(np.nan)
            pr_list.append(np.nan)

    def ci(arr):
        return np.nanpercentile(arr, [2.5, 97.5])

    return {
        "accuracy": acc_list,
        "f1": f1_list,
        "mcc": mcc_list,
        "roc_auc": roc_list,
        "pr_auc": pr_list
    }
    
def print_classification_results(name, results):
    print(f"\n{name}")
    print("-" * 40)

    print(f"Accuracy: {np.mean(results['accuracy']):.4f}")
    print(f"F1: {np.mean(results['f1']):.4f}")
    print(f"MCC: {np.mean(results['mcc']):.4f}")
    print(f"ROC-AUC: {np.mean(results['roc_auc']):.4f}")
    print(f"PR-AUC: {np.mean(results['pr_auc']):.4f}")


def format_results_classification(model, stage, results):
    return {
        "Model": model,
        "Stage": stage,
        "Accuracy": f"{np.mean(results['accuracy']):.2f}",
        "95% CI": f"[{np.percentile(results['accuracy'],2.5):.2f}, {np.percentile(results['accuracy'],97.5):.2f}]",
        "F1": f"{np.mean(results['f1']):.2f}",
        "MCC": f"{np.mean(results['mcc']):.2f}",
        "ROC-AUC": f"{np.mean(results['roc_auc']):.2f}",
        "Features": ""
    }

