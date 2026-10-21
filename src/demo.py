"""EduPredict AI: educational demonstration with synthetic data only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["attendance_pct", "average_grade", "assignment_completion_pct", "lms_activity_count", "previous_grade"]


def synthetic_students(n=700, seed=SEED):
    rng = np.random.default_rng(seed)
    attendance = 100 * rng.beta(6, 2, n)
    grades = np.clip(rng.normal(13, 3, n), 0, 20)
    completion = 100 * rng.beta(5, 2, n)
    lms = rng.poisson(25, n)
    previous = np.clip(grades + rng.normal(0, 2.5, n), 0, 20)
    # This arbitrary simulated mechanism is NOT a causal or empirical dropout model.
    logit = (-2.8 + 0.045 * (75 - attendance) + 0.25 * (12 - grades)
             + 0.035 * (70 - completion) + 0.025 * (20 - lms)
             + 0.08 * (12 - previous))
    probability = 1 / (1 + np.exp(-np.clip(logit, -40, 40)))
    label = rng.binomial(1, probability)
    return pd.DataFrame({"attendance_pct": attendance, "average_grade": grades,
                         "assignment_completion_pct": completion,
                         "lms_activity_count": lms, "previous_grade": previous,
                         "dropout": label})


def main():
    df = synthetic_students()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["dropout"], test_size=0.25,
        stratify=df["dropout"], random_state=SEED)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "k-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    }
    print("EDUPREDICT AI | SYNTHETIC DATA DEMONSTRATION")
    print(f"Synthetic students: {len(df)} | simulated positive label rate: {df['dropout'].mean():.1%}\n")
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        print(name)
        print(f"  Accuracy: {accuracy_score(y_test, pred):.3f}")
        print(f"  Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
        print(f"  Recall: {recall_score(y_test, pred, zero_division=0):.3f}")
        print(f"  F1: {f1_score(y_test, pred, zero_division=0):.3f}")
        print(f"  ROC AUC: {roc_auc_score(y_test, proba):.3f}")
        print(f"  Confusion matrix: {confusion_matrix(y_test, pred).tolist()}\n")
    print("Reminder: simulated outcomes; predictions are not valid for real student decisions.")


if __name__ == "__main__":
    main()
