"""
classical_models.py — KNN, Naive Bayes, Ensemble classifiers + evaluation suite.

Metrics: Accuracy, F1-score (macro), Confusion Matrix

Provides:
  - train_knn()              : KNN with multiple k values
  - train_naive_bayes()      : Gaussian + Multinomial NB
  - train_bagging()          : BaggingClassifier (Decision Tree base)
  - train_adaboost()         : AdaBoostClassifier
  - train_gradient_boosting(): GradientBoostingClassifier
  - grid_search_rf()         : GridSearchCV on RandomForest
  - grid_search_gbm()        : GridSearchCV on GradientBoosting
  - evaluate_classifier()    : unified evaluation (acc, F1, CM)
  - plot_confusion_matrix()  : heatmap
  - compare_models()         : bar chart comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import MinMaxScaler
import time
import json

from src import config


# ─────────────────────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, model_name="Model",
                         label_names=None, verbose=True):
    """
    Evaluate a fitted classifier. Returns dict with accuracy, f1, and confusion matrix.

    Args:
        model     : fitted sklearn classifier
        X_test    : (N, D) test features
        y_test    : (N,)   true labels
        model_name: display name
        label_names: list of class names

    Returns:
        results: dict {accuracy, f1_macro, f1_weighted, confusion_matrix, report}
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names,
                                   zero_division=0)
    results = {
        "model": model_name,
        "accuracy":    round(acc,    4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wt,  4),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    if verbose:
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  F1 (macro)  : {f1_mac:.4f}")
        print(f"  F1 (weighted): {f1_wt:.4f}")
        print(f"\n{report}")
    return results


def plot_confusion_matrix(cm, label_names=None, title="Confusion Matrix",
                          save_path=None, figsize=(9, 7)):
    """Plot normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names or "auto",
                yticklabels=label_names or "auto",
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(results_list, metric="accuracy", save_path=None):
    """
    Bar chart comparing multiple models on a given metric.

    Args:
        results_list: list of result dicts from evaluate_classifier()
        metric: 'accuracy' | 'f1_macro' | 'f1_weighted'
    """
    names  = [r["model"]  for r in results_list]
    values = [r[metric]   for r in results_list]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(max(6, len(names)*1.2), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, min(1.0, max(values) + 0.15))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Baseline Models
# ─────────────────────────────────────────────────────────────

def train_knn(X_train, y_train, X_test, y_test,
              k_values=None, label_names=None):
    """
    Train and evaluate KNN for multiple k values.

    Returns:
        best_model, best_results, all_results (list)
    """
    k_values = k_values or config.KNN_K_VALUES
    all_res  = []
    print(f"\n[KNN] Testing k = {k_values}")
    for k in k_values:
        t0    = time.time()
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        res = evaluate_classifier(model, X_test, y_test,
                                  model_name=f"KNN (k={k})",
                                  label_names=label_names, verbose=False)
        res["train_time"] = round(elapsed, 2)
        all_res.append((model, res))
        print(f"  k={k:2d} | Acc={res['accuracy']:.4f} | F1={res['f1_macro']:.4f} | t={elapsed:.1f}s")

    best_model, best_res = max(all_res, key=lambda x: x[1]["f1_macro"])
    print(f"\n  Best k: {best_res['model']} → F1={best_res['f1_macro']:.4f}")
    return best_model, best_res, [r for _, r in all_res]


def train_naive_bayes(X_train, y_train, X_test, y_test, label_names=None):
    """
    Train Gaussian NB and (if features non-negative) Multinomial NB.

    Returns:
        best_model, results_list
    """
    results = []
    # Gaussian NB — works with any features
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    res_g = evaluate_classifier(gnb, X_test, y_test,
                                model_name="GaussianNB", label_names=label_names)
    results.append((gnb, res_g))

    # Multinomial NB — needs non-negative features; scale to [0,1]
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_tr_scaled, y_train)
    res_m = evaluate_classifier(mnb, X_te_scaled, y_test,
                                model_name="MultinomialNB", label_names=label_names)
    results.append((mnb, res_m))

    best_model, _ = max(results, key=lambda x: x[1]["f1_macro"])
    return best_model, [r for _, r in results]


# ─────────────────────────────────────────────────────────────
# Ensemble Models
# ─────────────────────────────────────────────────────────────

def train_bagging(X_train, y_train, X_test, y_test,
                  n_estimators=50, label_names=None):
    """Bagging with Decision Tree base estimator."""
    print(f"\n[Bagging] n_estimators={n_estimators}")
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=n_estimators,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")
    return model, evaluate_classifier(model, X_test, y_test,
                                      model_name="BaggingClassifier",
                                      label_names=label_names)


def train_adaboost(X_train, y_train, X_test, y_test,
                   n_estimators=100, label_names=None):
    """AdaBoost with Decision Tree base (max_depth=2 — weak learners)."""
    print(f"\n[AdaBoost] n_estimators={n_estimators}")
    # Note: 'algorithm' param removed in scikit-learn 1.4 (SAMME is now the only algorithm)
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=n_estimators,
        learning_rate=0.5,
        random_state=config.RANDOM_SEED,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")
    return model, evaluate_classifier(model, X_test, y_test,
                                      model_name="AdaBoostClassifier",
                                      label_names=label_names)


def train_gradient_boosting(X_train, y_train, X_test, y_test,
                            n_estimators=100, learning_rate=0.1,
                            max_depth=3, label_names=None):
    """Gradient Boosting classifier."""
    print(f"\n[GBM] n_est={n_estimators} lr={learning_rate} depth={max_depth}")
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=config.RANDOM_SEED,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")
    return model, evaluate_classifier(model, X_test, y_test,
                                      model_name="GradientBoosting",
                                      label_names=label_names)


# ─────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────

def grid_search_rf(X_train, y_train, X_test, y_test,
                   param_grid=None, cv=3, label_names=None):
    """GridSearchCV on RandomForestClassifier."""
    param_grid = param_grid or config.RF_PARAM_GRID
    print(f"\n[GridSearch-RF] Grid: {param_grid}")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1),
        param_grid=param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1,
    )
    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f} | Time: {elapsed:.1f}s")
    res = evaluate_classifier(gs.best_estimator_, X_test, y_test,
                              model_name=f"RandomForest (best)", label_names=label_names)
    res["best_params"]   = gs.best_params_
    res["best_cv_score"] = round(gs.best_score_, 4)
    return gs.best_estimator_, res, gs


def grid_search_gbm(X_train, y_train, X_test, y_test,
                    param_grid=None, cv=3, label_names=None):
    """GridSearchCV on GradientBoostingClassifier."""
    param_grid = param_grid or config.GBM_PARAM_GRID
    print(f"\n[GridSearch-GBM] Grid: {param_grid}")
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=config.RANDOM_SEED),
        param_grid=param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1,
    )
    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f} | Time: {elapsed:.1f}s")
    res = evaluate_classifier(gs.best_estimator_, X_test, y_test,
                              model_name="GBM (best)", label_names=label_names)
    res["best_params"]   = gs.best_params_
    res["best_cv_score"] = round(gs.best_score_, 4)
    return gs.best_estimator_, res, gs


def save_results(results_list, save_path=None):
    """Save list of result dicts to JSON."""
    if save_path is None:
        save_path = config.REPORTS_DIR / "phase1_results.json"
    for r in results_list:
        if "confusion_matrix" in r and hasattr(r["confusion_matrix"], "tolist"):
            r["confusion_matrix"] = r["confusion_matrix"].tolist()
    with open(save_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"[Saved] Results → {save_path}")
