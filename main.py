from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Disease Prediction API")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD ML FILES =================
model = pickle.load(open("disease_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
all_symptoms = pickle.load(open("symptom_list.pkl", "rb"))

# ================= LOAD DATASET =================
data = pd.read_csv("datasetnew.csv").fillna("")
data = data.map(
    lambda x: x.strip().lower().replace(" ", "_") if isinstance(x, str) else x
)

symptom_columns = list(data.columns[1:])

# ================= RULE BASED MAP =================
disease_symptom_map = {}

for _, row in data.iterrows():
    disease = row["Disease"]
    symptoms = set([s for s in row[symptom_columns].values if s != ""])

    if disease in disease_symptom_map:
        disease_symptom_map[disease] |= symptoms
    else:
        disease_symptom_map[disease] = symptoms


# ================= INPUT MODEL =================
class PredictionRequest(BaseModel):
    symptoms: list[str]


@app.get("/")
def home():
    return {"message": "Disease Prediction API is running"}


@app.post("/predict")
def predict(data: PredictionRequest):

    # -------- clean symptoms --------
    user_symptoms = set(s.strip().lower().replace(" ", "_") for s in data.symptoms)

    if not user_symptoms:
        return {"error": "No symptoms provided"}

    # -------- ML PREDICTION --------
    input_vector = [1 if sym in user_symptoms else 0 for sym in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    ml_pred = model.predict(input_df)[0]
    ml_disease = label_encoder.inverse_transform([ml_pred])[0]

    # -------- RULE BASED PREDICTION (ONLY DISEASE NAMES) --------
    rule_results = []

    for disease, symptoms in disease_symptom_map.items():
        matched = len(user_symptoms & symptoms)
        if matched == 0:
            continue
        disease_name = disease.replace("_", " ").title()
        if disease_name != ml_disease.replace("_", " ").title():  # remove ML predicted disease
            rule_results.append(disease_name)

    # Optional: top 10 diseases
    rule_results = rule_results[:10]

    return {
        "ml_predict": ml_disease.replace("_", " ").title(),
        "rule_based_predict": rule_results
    }

