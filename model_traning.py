import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ================= LOAD DATA =================
data = pd.read_csv("datasetnew.csv").fillna("")

def clean_text(x):
    if isinstance(x, str):
        return x.strip().lower().replace(" ", "_")
    return x

data = data.applymap(clean_text)

# ================= SYMPTOM COLUMNS =================
symptom_columns = [c for c in data.columns if c.lower().startswith("symptom")]

# ================= ALL UNIQUE SYMPTOMS =================
all_symptoms = sorted({
    sym for row in data[symptom_columns].values
    for sym in row if sym != ""
})

# ================= MULTI-HOT ENCODING =================
def encode_row(row):
    present = set(s for s in row if s != "")
    return [1 if sym in present else 0 for sym in all_symptoms]

X = pd.DataFrame(
    [encode_row(row) for row in data[symptom_columns].values],
    columns=all_symptoms
)

# ================= LABEL ENCODING =================
y = data["Disease"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ================= SMOTE (IMBALANCE FIX) =================
smote = SMOTE(random_state=42, k_neighbors=2)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# ================= TRAIN–TEST SPLIT =================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for train_idx, test_idx in sss.split(X_balanced, y_balanced):
    X_train, X_test = X_balanced.iloc[train_idx], X_balanced.iloc[test_idx]
    y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

# ================= MODEL =================
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=2,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("=" * 50)
print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
print("=" * 50)

# ================= SAVE FILES =================
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(all_symptoms, open("symptom_list.pkl", "wb"))

print("✅ Saved model & encoders successfully")
