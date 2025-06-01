
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import traceback

app = FastAPI()

# Model ve scaler'ı yükle
model_package = load("physics_rf_full_model_last_v2.pkl")
model = model_package["model"]
scaler = model_package["scaler"]

class UserData(BaseModel):
    gender: int
    part_time_job: int
    absence_days: int
    extracurricular_activities: int
    weekly_self_study_hours: float
    math_score: float

@app.post("/predict")
async def predict(data: UserData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        score = prediction[0]

        recommendations = []

        if data.part_time_job == 1:
            recommendations.append("Part-time çalışmanın ders başarına olumsuz etkisi olabilir.")
        else:
            recommendations.append("Part-time işin yok, derslerine daha fazla odaklanabilirsin.")

        if data.weekly_self_study_hours < 5:
            recommendations.append("Haftalık çalışma süren düşük.")
        elif data.weekly_self_study_hours >= 10:
            recommendations.append("Çalışma süren mükemmel seviyede.")
        else:
            recommendations.append("Çalışma süren fena değil ama biraz daha artırabilirsin.")

        if data.absence_days > 5:
            recommendations.append("Devamsızlık sayın yüksek.")
        else:
            recommendations.append("Devamsızlık durumun iyi.")

        if data.extracurricular_activities == 1:
            recommendations.append("Ders dışı etkinliklere katılman güzel.")
        else:
            recommendations.append("Etkinliklere katılmıyorsan, sosyal aktiviteler motivasyonunu artırabilir.")

        if score >= 90:
            motivation = "Harikasın!"
        elif 70 <= score < 90:
            motivation = "Çok iyi gidiyorsun!"
        elif 50 <= score < 70:
            motivation = "Fena değil."
        else:
            motivation = "Daha yolun başındasın."

        general_tip = "Düzenli tekrar ve bol soru çözümü çok önemli."

        full_recommendation = " ".join(recommendations)

        response_data = {
            "predicted_score": float(score),
            "recommendation": full_recommendation,
            "motivation": motivation,
            "general_tip": general_tip
        }

        return response_data

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "API çalışıyor!"}
