
# ============================================================
# MSA-GCRNN Framework for Smart Mobility Traffic Forecasting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare

import shap
import lime
import lime.lime_tabular

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = "smart_mobility_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

cyber_physical_features = [
    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_%",
    "Traffic_Light_State",
    "Weather_Condition",
    "Accident_Report",
    "Emission_Levels_g_km",
    "Energy_Consumption_L_h",
    "Latitude",
    "Longitude"
]

social_features = [
    "Sentiment_Score",
    "Ride_Sharing_Demand",
    "Parking_Availability",
    "Public_Transport_Delay"
]

target_column = "Traffic_Condition"

all_features = cyber_physical_features + social_features

# ============================================================
# PREPROCESSING
# ============================================================

for col in all_features:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.replace(r"[^0-9a-zA-Z._ -]", "", regex=True)

categorical_cols = df.select_dtypes(include="object").columns.tolist()

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

imputer = KNNImputer(n_neighbors=5)
df[all_features] = imputer.fit_transform(df[all_features])

scaler = MinMaxScaler()
df[all_features] = scaler.fit_transform(df[all_features])

# ============================================================
# EDA
# ============================================================

plt.figure(figsize=(8,5))
sns.boxplot(x=df[target_column], y=df["Traffic_Speed_kmh"])
plt.title("Traffic Speed Distribution")
plt.tight_layout()
plt.savefig("eda_speed_distribution.png", dpi=300)

corr = df[all_features].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

X_full = df[all_features]
y_full = df[target_column]

mi_scores = mutual_info_classif(X_full, y_full)

mi_df = pd.DataFrame({
    "Feature": all_features,
    "MI_Score": mi_scores
}).sort_values(by="MI_Score", ascending=False)

print(mi_df)

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

feature_similarity = cosine_similarity(X_full)

coords = df[["Latitude", "Longitude"]].values

distance_matrix = np.linalg.norm(
    coords[:, None, :] - coords[None, :, :],
    axis=2
)

sigma = np.mean(distance_matrix[distance_matrix > 0])

spatial_similarity = np.exp(
    -(distance_matrix ** 2) / (2 * sigma ** 2)
)

alpha = 0.7

adjacency_matrix = (
    alpha * feature_similarity +
    (1 - alpha) * spatial_similarity
)

np.fill_diagonal(adjacency_matrix, 1)

# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

X_train = train_df[all_features].values
y_train = train_df[target_column].values

X_val = val_df[all_features].values
y_val = val_df[target_column].values

X_test = test_df[all_features].values
y_test = test_df[target_column].values

# ============================================================
# BASELINE MODELS
# ============================================================

baseline_models = {
    "SVM": SVC(probability=True),
    "LR": LogisticRegression(max_iter=1000),
    "DT": DecisionTreeClassifier(max_depth=10),
    "RF": RandomForestClassifier(n_estimators=100),
    "XGB": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        eval_metric="mlogloss"
    )
}

results = []

for name, model in baseline_models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    results.append([name, acc, pre, rec, f1])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print(results_df)

# ============================================================
# MSA-GCRNN MODEL
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):

        attn_output, _ = self.attention(x, x, x)

        return attn_output

class MSA_GCRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_dim,
            128,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            128,
            256,
            batch_first=True
        )

        self.attention = MultiHeadAttention(
            hidden_dim=256,
            heads=8
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):

        out, _ = self.lstm1(x)

        out, _ = self.lstm2(out)

        out = self.attention(out)

        out = self.dropout(out)

        out = out[:, -1, :]

        out = self.fc(out)

        return out

# ============================================================
# SEQUENCE GENERATION
# ============================================================

SEQ_LEN = 30

def create_sequences(X, y, seq_len):

    xs = []
    ys = []

    for i in range(len(X) - seq_len):

        x_seq = X[i:i+seq_len]
        y_seq = y[i+seq_len]

        xs.append(x_seq)
        ys.append(y_seq)

    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)

# ============================================================
# TRAINING
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSA_GCRNN(
    input_dim=len(all_features),
    hidden_dim=256,
    output_dim=len(np.unique(y_full))
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5
)

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train_tensor.to(device))

    loss = criterion(outputs, y_train_tensor.to(device))

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val_tensor.to(device))

        val_loss = criterion(
            val_outputs,
            y_val_tensor.to(device)
        )

    scheduler.step(val_loss)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        torch.save(
            model.state_dict(),
            "best_msa_gcrnn.pth"
        )

    print(
        f"Epoch {epoch+1}/100 "
        f"Train Loss: {loss.item():.4f} "
        f"Val Loss: {val_loss.item():.4f}"
    )

# ============================================================
# TESTING
# ============================================================

model.load_state_dict(
    torch.load("best_msa_gcrnn.pth")
)

model.eval()

with torch.no_grad():

    logits = model(X_test_tensor.to(device))

    predictions = torch.argmax(logits, dim=1).cpu().numpy()

acc = accuracy_score(y_test_seq, predictions)
pre = precision_score(y_test_seq, predictions, average="weighted")
rec = recall_score(y_test_seq, predictions, average="weighted")
f1 = f1_score(y_test_seq, predictions, average="weighted")

print("\nMSA-GCRNN Performance")
print("Accuracy:", acc)
print("Precision:", pre)
print("Recall:", rec)
print("F1:", f1)

# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

bootstrap_scores = []

n_iterations = 1000

for i in range(n_iterations):

    indices = np.random.choice(
        len(y_test_seq),
        len(y_test_seq),
        replace=True
    )

    score = accuracy_score(
        y_test_seq[indices],
        predictions[indices]
    )

    bootstrap_scores.append(score)

lower = np.percentile(bootstrap_scores, 2.5)
upper = np.percentile(bootstrap_scores, 97.5)

print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

# ============================================================
# STATISTICAL TESTING
# ============================================================

baseline_acc = results_df["Accuracy"].values

proposed_scores = np.repeat(acc, len(baseline_acc))

t_stat, p_val = ttest_rel(proposed_scores, baseline_acc)

print("Paired t-test:", t_stat, p_val)

try:
    w_stat, w_p = wilcoxon(proposed_scores, baseline_acc)
    print("Wilcoxon:", w_stat, w_p)
except:
    pass

# ============================================================
# SHAP ANALYSIS
# ============================================================

rf_model = baseline_models["RF"]

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=all_features,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)

# ============================================================
# LIME ANALYSIS
# ============================================================

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=all_features,
    class_names=[str(c) for c in np.unique(y_full)],
    discretize_continuous=True
)

exp = lime_explainer.explain_instance(
    X_test[0],
    rf_model.predict_proba,
    num_features=10
)

fig = exp.as_pyplot_figure()

plt.tight_layout()
plt.savefig("lime_explanation.png", dpi=300)

# ============================================================
# ROC CURVES
# ============================================================

plt.figure(figsize=(8,6))

for name, model in baseline_models.items():

    probs = model.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(
        y_test,
        probs[:, 1],
        pos_label=1
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={roc_auc:.2f}"
    )

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)

# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)

# ============================================================
# K-FOLD VALIDATION
# ============================================================

kf = KFold(
    n_splits=7,
    shuffle=False
)

cv_scores = []

for train_idx, test_idx in kf.split(X_full):

    X_tr = X_full.iloc[train_idx]
    X_te = X_full.iloc[test_idx]

    y_tr = y_full.iloc[train_idx]
    y_te = y_full.iloc[test_idx]

    model_cv = RandomForestClassifier()

    model_cv.fit(X_tr, y_tr)

    preds_cv = model_cv.predict(X_te)

    score = accuracy_score(y_te, preds_cv)

    cv_scores.append(score)

print("Cross Validation Scores:", cv_scores)
print("Mean CV:", np.mean(cv_scores))

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "baseline_results.csv",
    index=False
)

print("\nPipeline Completed Successfully.")
