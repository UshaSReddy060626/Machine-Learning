import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,)
def main():
    # 1. Load and inspect the data
    df = pd.read_csv("diabetes.csv")
    print("Shape:", df.shape)
    print(df.head(), "\n")
    print(df.describe(), "\n")
    # 2. Replace invalid zeros with NaN
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
    print("Missing values after replacing invalid zeros:")
    print(df.isnull().sum(), "\n")
    # 3. Train-test split
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 4. Median imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    # 6. Train balanced Logistic Regression
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    # 7. Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("--- Evaluation Metrics ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    # Confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})", color="#2c7fb8")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()
    # 8. Interpret coefficients
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", ascending=False)
    print("\n--- Coefficients (sorted) ---")
    print(coef_df)
    plt.figure(figsize=(7, 5))
    colors = ["#2c7fb8" if c > 0 else "#de2d26" for c in coef_df["Coefficient"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (standardized)")
    plt.title("Logistic Regression Coefficients")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("coefficients.png", dpi=150)
    plt.close()
    print("\nKey Findings:")
    print(f"- ROC-AUC ~ {roc_auc:.3f}: good class-separation ability.")
    print(f"- Recall = {rec:.3f}: ~{rec*100:.1f}% of actual diabetic cases correctly identified.")
    print("- Glucose and BMI have the strongest positive coefficients.")
    print("\nLimitation: Small historical dataset representing a specific population.")
    print("Suitable for learning classification concepts, not for medical diagnosis.")
if __name__ == "__main__":
    main()