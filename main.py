import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


df = pd.read_csv("data/oasis_longitudinal.csv")

print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDiagnosis distribution:\n", df["Group"].value_counts())



data = df[df["Group"].isin(["Demented", "Nondemented"])].copy()

features = [
    "M/F", "Age", "EDUC", "SES",
    "MMSE", "eTIV", "nWBV", "ASF"
]

X = data[features]
y = data["Group"].map({"Nondemented": 0, "Demented": 1})

numeric_features = [
    "Age", "EDUC", "SES", "MMSE",
    "eTIV", "nWBV", "ASF"
]

categorical_features = ["M/F"]


preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

scaled_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])



splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=data["Subject ID"])
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

groups_train = data.iloc[train_idx]["Subject ID"]



models = {
    "Random Forest": Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("preprocessing", scaled_preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("preprocessing", scaled_preprocessor),
        ("classifier", SVC(kernel="rbf"))
    ])
}



cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n===== CROSS-VALIDATION RESULTS =====")

for name, candidate in models.items():
    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="balanced_accuracy"
    )

    print(f"\n{name}")
    print(f"Mean balanced accuracy: {scores.mean():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")



rf_model = models["Random Forest"]

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\n===== RANDOM FOREST TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, rf_predictions))

print(classification_report(
    y_test,
    rf_predictions,
    target_names=["Nondemented", "Demented"],
    zero_division=0
))



lr_model = models["Logistic Regression"]

lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("\n===== LOGISTIC REGRESSION TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, lr_predictions))

print(classification_report(
    y_test,
    lr_predictions,
    target_names=["Nondemented", "Demented"],
    zero_division=0
))


fig1, ax1 = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=["Nondemented", "Demented"],
    cmap="Blues",
    ax=ax1
)

ax1.set_title("Random Forest - Confusion Matrix")
fig1.tight_layout()


fig2, ax2 = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    lr_predictions,
    display_labels=["Nondemented", "Demented"],
    cmap="Greens",
    ax=ax2
)

ax2.set_title("Logistic Regression - Confusion Matrix")
fig2.tight_layout()



importance = permutation_importance(
    lr_model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="balanced_accuracy"
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n FEATURE IMPORTANCE ")
print(importance_df)


fig3, ax3 = plt.subplots(figsize=(9, 5))

ax3.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    xerr=importance_df["Std"],
    color="steelblue"
)

ax3.invert_yaxis()

ax3.set_xlabel("Mean Decrease in Balanced Accuracy")
ax3.set_title("Logistic Regression - Permutation Importance")

fig3.tight_layout()


param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    estimator=models["Logistic Regression"],
    param_grid=param_grid,
    cv=cv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
)

grid_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\n HYPERPARAMETER TUNING ")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV balanced accuracy: {grid_search.best_score_:.3f}")

tuned_model = grid_search.best_estimator_


Path("models").mkdir(exist_ok=True)

joblib.dump(
    tuned_model,
    "models/logistic_regression.joblib"
)

print("\nTuned model saved successfully!")






plt.show()
