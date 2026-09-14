
# Module 2 — Analytics Pipeline
# Can be run in Google Colab or locally.
# Raw Titanic data is loaded from Seaborn exactly once, then saved to titanic.csv.
# All later stages use the cleaned DataFrame / saved CSV.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

RANDOM_STATE = 42
os.makedirs("charts", exist_ok=True)

# ============================================================
# 1. LOAD ONCE + PROFILE + OFFLINE FALLBACK
# ============================================================
df = sns.load_dataset("titanic")  # ONLY raw network/cache load in this module

print("=== DATA INFO ===")
df.info()

print("\n=== DESCRIBE ===")
print(df.describe(include="all"))

print("\n=== SHAPE ===")
print(df.shape)

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

print("\n=== MISSING VALUES (%) ===")
print(missing_pct)

# Required raw fallback immediately after loading.
df.to_csv("titanic.csv", index=False)
print("\nSaved offline fallback: titanic.csv")

# ============================================================
# 2. CLEANING — THRESHOLD RULE
# ============================================================
clean_df = df.copy()

print("\n=== CLEANING DECISIONS ===")
for col, pct in missing_pct.items():
    if pct < 5:
        print(f"{col}: {pct:.2f}% missing -> DROP ROWS (<5%).")
    elif pct <= 30:
        print(f"{col}: {pct:.2f}% missing -> IMPUTE (5%-30%).")
    else:
        print(
            f"{col}: {pct:.2f}% missing -> ENCODE MISSING AS CATEGORY "
            f"(>30%; preserves information and avoids unreliable imputation)."
        )

# Under 5%: drop rows with missing values in those columns.
low_missing_cols = [c for c, p in missing_pct.items() if p < 5]
if low_missing_cols:
    clean_df = clean_df.dropna(subset=low_missing_cols)

# 5%-30%: median for numeric, mode for categorical.
mid_missing_cols = [c for c, p in missing_pct.items() if 5 <= p <= 30]
for col in mid_missing_cols:
    if pd.api.types.is_numeric_dtype(clean_df[col]):
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    else:
        mode = clean_df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        clean_df[col] = clean_df[col].fillna(fill_value)

# >30%: explicit missing category.
high_missing_cols = [c for c, p in missing_pct.items() if p > 30]
for col in high_missing_cols:
    if pd.api.types.is_categorical_dtype(clean_df[col]):
        if "Missing" not in clean_df[col].cat.categories:
            clean_df[col] = clean_df[col].cat.add_categories(["Missing"])
    clean_df[col] = clean_df[col].fillna("Missing")

print("\nRemaining missing values:")
print(clean_df.isna().sum()[clean_df.isna().sum() > 0])

# For EDA, use a copy with numeric NaNs removed only where a statistic requires it.
# The modeling pipeline below independently performs train-only preprocessing.

# ============================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================
for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(clean_df[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"{col.title()} Histogram")
    sns.boxplot(x=clean_df[col].dropna(), ax=axes[1])
    axes[1].set_title(f"{col.title()} Box Plot")
    plt.tight_layout()
    plt.savefig(f"charts/{col}_univariate.png", dpi=150)
    plt.show()
    plt.close()

def iqr_outlier_count(series):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum()), q1, q3, lower, upper

for col in ["age", "fare"]:
    count, q1, q3, lower, upper = iqr_outlier_count(clean_df[col])
    print(
        f"\n{col}: Q1={q1:.4f}, Q3={q3:.4f}, "
        f"lower={lower:.4f}, upper={upper:.4f}, IQR outliers={count}"
    )

fare = clean_df["fare"].dropna()
fare_mean = fare.mean()
fare_median = fare.median()
fare_mode = fare.mode().iloc[0]

print("\n=== FARE STATISTICS ===")
print(f"Mean   : {fare_mean:.4f}")
print(f"Median : {fare_median:.4f}")
print(f"Mode   : {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    skew_text = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_text = "left-skewed"
else:
    skew_text = "not strictly classified by mean/median/mode ordering alone"

print("Fare distribution:", skew_text)

# ============================================================
# 4. BIVARIATE ANALYSIS
# ============================================================
print("\n=== SURVIVAL RATE BY SEX ===")
sex_survival = clean_df.groupby("sex", observed=True)["survived"].mean()
print((sex_survival * 100).round(2))

print("\n=== SURVIVAL RATE BY PCLASS ===")
pclass_survival = clean_df.groupby("pclass", observed=True)["survived"].mean()
print((pclass_survival * 100).round(2))

print("\n=== SURVIVAL RATE BY SEX + PCLASS ===")
sex_pclass_survival = (
    clean_df.groupby(["sex", "pclass"], observed=True)["survived"].mean()
)
print((sex_pclass_survival * 100).round(2))

# Boolean masking explicitly used.
female_first_or_second = clean_df[
    (clean_df["sex"] == "female") & (clean_df["pclass"].isin([1, 2]))
]
male_third = clean_df[
    (clean_df["sex"] == "male") & (clean_df["pclass"] == 3)
]
print(
    "\nBoolean-mask example:",
    f"female + class 1/2 survival={female_first_or_second['survived'].mean()*100:.2f}%",
    f"| male + class 3 survival={male_third['survived'].mean()*100:.2f}%"
)

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean_df[corr_cols].corr()

print("\n=== EXACT 6-COLUMN CORRELATION MATRIX ===")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Titanic Correlation Heatmap — Required Six Columns")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=150)
plt.show()
plt.close()

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append((abs(corr.iloc[i, j]), corr.iloc[i, j], corr_cols[i], corr_cols[j]))
pairs = sorted(pairs, reverse=True)

print("\n=== TWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS ===")
for rank, (abs_r, r, a, b) in enumerate(pairs[:2], start=1):
    print(f"{rank}. {a} vs {b}: r={r:.4f}, |r|={abs_r:.4f}")

# ============================================================
# 5. MULTIVARIATE DATA STORY — 4+ DISTINCT CHARTS
# ============================================================

# Chart 1: survival by sex
plt.figure(figsize=(7, 5))
sns.barplot(data=clean_df, x="sex", y="survived", errorbar=None)
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex")
plt.tight_layout()
plt.savefig("charts/01_survival_by_sex.png", dpi=150)
plt.show()
plt.close()

print("""
Interpretation — Chart 1:
Survival differs substantially by sex, with female passengers having a much
higher survival rate than male passengers. This indicates that sex is a strong
predictive feature for the Titanic survival outcome.
""")

# Chart 2: survival by passenger class and sex
plt.figure(figsize=(8, 5))
sns.barplot(data=clean_df, x="pclass", y="survived", hue="sex", errorbar=None)
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class and Sex")
plt.tight_layout()
plt.savefig("charts/02_survival_by_class_sex.png", dpi=150)
plt.show()
plt.close()

print("""
Interpretation — Chart 2:
Passenger class changes survival probability for both sexes, with higher-class
passengers generally having better outcomes. The combined sex/class view shows
that survival was influenced by both demographic and socioeconomic position.
""")

# Chart 3: age distribution by survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=clean_df, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig("charts/03_age_by_survival.png", dpi=150)
plt.show()
plt.close()

print("""
Interpretation — Chart 3:
The age distributions of survivors and non-survivors overlap, but their
central tendencies and spread are not identical. Age therefore provides
additional information beyond sex and passenger class, although it is not
sufficient by itself to explain survival.
""")

# Chart 4: fare vs age, colored by survival
plt.figure(figsize=(9, 6))
sns.scatterplot(data=clean_df, x="age", y="fare", hue="survived", alpha=0.65)
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig("charts/04_age_fare_survival.png", dpi=150)
plt.show()
plt.close()

print("""
Interpretation — Chart 4:
Fare is generally higher for some passengers in the upper classes, while age
spans a broad range in both survival groups. The plot supports the idea that
economic position, represented partly by fare, interacts with other features
rather than acting as a standalone explanation.
""")

# Chart 5: survival heatmap by sex/class
pivot = clean_df.pivot_table(
    index="sex", columns="pclass", values="survived", aggfunc="mean"
)
plt.figure(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Survival Rate Heatmap: Sex × Passenger Class")
plt.tight_layout()
plt.savefig("charts/05_survival_heatmap.png", dpi=150)
plt.show()
plt.close()

print("""
Interpretation — Chart 5:
The sex-by-class heatmap makes the interaction especially clear: survival rates
vary across both dimensions. It reinforces the overall data story that sex and
passenger class together provide a strong explanation of who was more likely
to survive.
""")

# ============================================================
# 6. EDA-ONLY STANDARDIZATION CHECK
# ============================================================
eda_scaled = clean_df[["age", "fare"]].copy()
before = eda_scaled.agg(["mean", "std"])

for col in ["age", "fare"]:
    eda_scaled[col] = (eda_scaled[col] - eda_scaled[col].mean()) / eda_scaled[col].std()

after = eda_scaled.agg(["mean", "std"])

print("\n=== STANDARDIZATION CHECK ===")
print("Before:")
print(before)
print("\nAfter:")
print(after)

# This is EDA only and is NOT fed into modeling.

# ============================================================
# 7. STRATIFIED TRAIN/TEST SPLIT FIRST
# ============================================================
# Classification modeling features.
model_df = clean_df.copy()

target = "survived"
feature_cols = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "embarked", "class", "who", "adult_male", "deck",
    "embark_town", "alone"
]

X = model_df[feature_cols].copy()
y = model_df[target].copy()

print("\n=== CLASS BALANCE ===")
print(y.value_counts())
print((y.value_counts(normalize=True) * 100).round(2))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Train class %:")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("Test class %:")
print((y_test.value_counts(normalize=True) * 100).round(2))

# Stratification keeps the class proportions similar in train and test,
# which is important because survival is a binary classification target.

# ============================================================
# 8. TRAIN-ONLY PREPROCESSING
# ============================================================
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = [
    "sex", "embarked", "class", "who", "adult_male",
    "deck", "embark_town", "alone"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================================
# 9. THREE CLASSIFIERS — SAME SPLIT
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    )
}

fitted_models = {}
results = {}
roc_data = {}

for name, estimator in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    results[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }
    roc_data[name] = (prob, auc)

    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    # Separate confusion matrix image for each model.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"charts/confusion_{safe_name}.png", dpi=150)
    plt.show()
    plt.close()

# ROC curve for all three.
plt.figure(figsize=(8, 6))
for name, (prob, auc) in roc_data.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Three Classifiers")
plt.legend()
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150)
plt.show()
plt.close()

classification_table = pd.DataFrame(results).T
print("\n=== CLASSIFIER COMPARISON ===")
print(classification_table.round(4))
classification_table.to_csv("classifier_comparison.csv")

# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================
tree_pipe = fitted_models["Decision Tree"]
tree_pre = tree_pipe.named_steps["preprocessor"]
tree_model = tree_pipe.named_steps["model"]

feature_names = tree_pre.get_feature_names_out()

plt.figure(figsize=(24, 14))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=False,
    rounded=True,
    max_depth=4,
    fontsize=7
)
plt.title("Decision Tree (first 4 levels shown)")
plt.tight_layout()
plt.savefig("charts/decision_tree.png", dpi=180)
plt.show()
plt.close()

# ============================================================
# 11. IMBALANCE COMPARISON — BASELINE / BALANCED / SMOTE
# ============================================================
# SMOTE is optional in the environment. Install imbalanced-learn in Colab
# if necessary. The try/except gives a clear instruction.
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

imbalance_results = {}

# Use Random Forest for this sub-task.
base_rf = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_STATE
)

balanced_rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
)

for variant, estimator in [
    ("Baseline", base_rf),
    ("Class Weight Balanced", balanced_rf)
]:
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    imbalance_results[variant] = {
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    }

if SMOTE_AVAILABLE:
    smote_pipe = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE
            ))
        ]
    )
    smote_pipe.fit(X_train, y_train)
    smote_pred = smote_pipe.predict(X_test)

    imbalance_results["SMOTE"] = {
        "Precision": precision_score(y_test, smote_pred, zero_division=0),
        "Recall": recall_score(y_test, smote_pred, zero_division=0),
        "F1": f1_score(y_test, smote_pred, zero_division=0)
    }
else:
    print("\nSMOTE is not installed.")
    print("In Google Colab run: !pip -q install imbalanced-learn")
    print("Then rerun the imbalance section.")

imbalance_table = pd.DataFrame(imbalance_results).T
print("\n=== IMBALANCE COMPARISON ===")
print(imbalance_table.round(4))
imbalance_table.to_csv("imbalance_comparison.csv")

if not imbalance_table.empty:
    best_imbalance = imbalance_table["F1"].idxmax()
    print(
        f"\nImbalance conclusion: {best_imbalance} has the highest observed F1 "
        "on the held-out test set. Selection should also consider whether "
        "precision or recall is more important for the intended use case."
    )

# ============================================================
# 12. RANDOM FOREST GRID SEARCH + OOB SCORE
# ============================================================
# GridSearchCV is performed on the train split. OOB is then evaluated on
# a refitted RandomForest with the selected parameters and oob_score=True.
rf_for_grid = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            oob_score=True
        ))
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=rf_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("\n=== GRID SEARCH ===")
print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_params_model = {
    key.replace("model__", ""): value
    for key, value in grid.best_params_.items()
}

oob_rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    oob_score=True,
    **best_params_model
)

oob_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", oob_rf)
    ]
)
oob_pipeline.fit(X_train, y_train)

print("OOB score:", oob_pipeline.named_steps["model"].oob_score_)

# ============================================================
# 13. REGRESSION SIDE TASK — PREDICT FARE
# ============================================================
# Use the same cleaned dataset. Split first, then train-only preprocessing.
regression_target = "fare"
regression_features = [
    c for c in clean_df.columns
    if c != regression_target
    and c != "survived"
]

reg_df = clean_df[regression_features + [regression_target]].copy()

# Drop rows with missing regression target; features are handled in pipeline.
reg_df = reg_df.dropna(subset=[regression_target])

X_reg = reg_df[regression_features]
y_reg = reg_df[regression_target]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)

reg_num = Xr_train.select_dtypes(include=["number"]).columns.tolist()
reg_cat = Xr_train.select_dtypes(exclude=["number"]).columns.tolist()

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            reg_num
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            reg_cat
        )
    ]
)

reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ]
)

reg_pipeline.fit(Xr_train, yr_train)
yr_pred = reg_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)

n = len(yr_test)
p = reg_pipeline.named_steps["preprocessor"].transform(Xr_test).shape[1]
if n - p - 1 > 0:
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
else:
    adjusted_r2 = np.nan

print("\n=== REGRESSION METRICS ===")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R2        : {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(8, 5))
plt.scatter(yr_pred, residuals, alpha=0.65)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig("charts/regression_residuals.png", dpi=150)
plt.show()
plt.close()

# Simple numerical check to support the written conclusion.
# Heteroscedasticity is suggested when residual spread changes substantially
# across fitted values; visual inspection remains important.
bins = pd.qcut(pd.Series(yr_pred), q=4, duplicates="drop")
resid_std_by_bin = pd.Series(residuals).groupby(bins, observed=True).std()
spread_ratio = (
    resid_std_by_bin.max() / resid_std_by_bin.min()
    if len(resid_std_by_bin) > 1 and resid_std_by_bin.min() > 0
    else np.nan
)
hetero = bool(pd.notna(spread_ratio) and spread_ratio > 1.5)

print("\nResidual spread by predicted-fare quartile:")
print(resid_std_by_bin)
print(
    "Heteroscedasticity conclusion:",
    "The residual spread changes materially across predicted values, suggesting heteroscedasticity."
    if hetero
    else "The residual spread is not strongly changing across predicted values; strong heteroscedasticity is not evident."
)

# ============================================================
# 14. FINAL MODEL COMPARISON TABLE + RECOMMENDATION
# ============================================================
final_table = classification_table.copy()
final_table["MAE"] = np.nan
final_table["RMSE"] = np.nan
final_table["R2"] = np.nan
final_table["Adjusted R2"] = np.nan

# Regression is a separate metric group; it is not comparable directly
# to classification accuracy/precision/recall/F1/AUC.
regression_row = pd.DataFrame(
    {
        "Accuracy": [np.nan],
        "Precision": [np.nan],
        "Recall": [np.nan],
        "F1": [np.nan],
        "AUC": [np.nan],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2],
        "Adjusted R2": [adjusted_r2]
    },
    index=["Linear Regression — Fare"]
)

final_table = pd.concat([final_table, regression_row])
print("\n=== FINAL MODEL COMPARISON ===")
print(final_table.round(4))
final_table.to_csv("final_model_comparison.csv")

best_classifier = classification_table["F1"].idxmax()
best_f1 = classification_table.loc[best_classifier, "F1"]
best_auc = classification_table.loc[best_classifier, "AUC"]
best_acc = classification_table.loc[best_classifier, "Accuracy"]

recommendation = f"""
Final recommendation:
I would deploy {best_classifier} for the Titanic survival classification task
because it achieved the highest F1 score among the three evaluated classifiers
(F1={best_f1:.4f}), while also achieving accuracy={best_acc:.4f} and
AUC={best_auc:.4f} on the same held-out test set. F1 balances precision and
recall, while AUC summarizes ranking performance across classification
thresholds. The final decision should also consider business costs of false
positives versus false negatives; based strictly on the required evaluation
metrics, {best_classifier} is the strongest choice in this experiment.
"""

print(recommendation)

# ============================================================
# 15. SAVE BEST COMPLETE END-TO-END PIPELINE
# ============================================================
# Choose best classifier by F1 and refit its COMPLETE pipeline on training data.
best_estimators = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    )
}

final_classifier = best_estimators[best_classifier]

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", final_classifier)
    ]
)

full_pipeline.fit(X_train, y_train)

joblib.dump(full_pipeline, "best_titanic_pipeline.joblib")
print("\nSaved complete pipeline: best_titanic_pipeline.joblib")

# Reload and confirm raw-data prediction works.
loaded_pipeline = joblib.load("best_titanic_pipeline.joblib")
raw_sample = X_test.iloc[[0]]
loaded_prediction = loaded_pipeline.predict(raw_sample)[0]
loaded_probability = loaded_pipeline.predict_proba(raw_sample)[0, 1]

print("\n=== RELOAD TEST ===")
print("Raw input:")
print(raw_sample)
print("Reloaded pipeline prediction:", loaded_prediction)
print("Reloaded pipeline survival probability:", round(loaded_probability, 4))

# ============================================================
# MODULE SUMMARY FILES
# ============================================================
with open("query_output.txt", "w", encoding="utf-8") as f:
    f.write("MODULE 2 — ANALYTICS PIPELINE OUTPUT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Original shape: {df.shape}\n")
    f.write(f"Cleaned shape : {clean_df.shape}\n\n")
    f.write("Missing percentages:\n")
    f.write(missing_pct.to_string() + "\n\n")
    f.write("IQR outliers:\n")
    for col in ["age", "fare"]:
        count, q1, q3, lower, upper = iqr_outlier_count(clean_df[col])
        f.write(f"{col}: {count}\n")
    f.write("\nFare statistics:\n")
    f.write(f"Mean={fare_mean:.4f}\nMedian={fare_median:.4f}\nMode={fare_mode:.4f}\n")
    f.write(f"Skewness conclusion={skew_text}\n\n")
    f.write("Survival by sex (%):\n")
    f.write((sex_survival * 100).round(2).to_string() + "\n\n")
    f.write("Survival by pclass (%):\n")
    f.write((pclass_survival * 100).round(2).to_string() + "\n\n")
    f.write("Survival by sex+pclass (%):\n")
    f.write((sex_pclass_survival * 100).round(2).to_string() + "\n\n")
    f.write("Correlation matrix:\n")
    f.write(corr.round(4).to_string() + "\n\n")
    f.write("Two strongest correlations:\n")
    for rank, (abs_r, r, a, b) in enumerate(pairs[:2], start=1):
        f.write(f"{rank}. {a} vs {b}: r={r:.4f}, |r|={abs_r:.4f}\n")
    f.write("\nClassifier metrics:\n")
    f.write(classification_table.round(4).to_string() + "\n\n")
    f.write("Imbalance comparison:\n")
    f.write(imbalance_table.round(4).to_string() + "\n\n")
    f.write("Grid best parameters:\n")
    f.write(str(grid.best_params__) + "\n")
    f.write(f"OOB score: {oob_pipeline.named_steps['model'].oob_score_:.4f}\n\n")
    f.write("Regression metrics:\n")
    f.write(f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}\nAdjusted R2={adjusted_r2:.4f}\n")
    f.write("\n")
    f.write(recommendation)

print("\n=== MODULE 2 COMPLETE ===")
print("Generated:")
print("- titanic.csv")
print("- best_titanic_pipeline.joblib")
print("- query_output.txt")
print("- classifier_comparison.csv")
print("- imbalance_comparison.csv")
print("- final_model_comparison.csv")
print("- charts/*.png")
