from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
from io import BytesIO
import pickle

app = FastAPI()
with open("models/reference_model.pkl", "rb") as f:
    reference_model = pickle.load(f)
with open("models/balanced_model.pkl", "rb") as f:
    balanced_model = pickle.load(f)
with open("models/best_tree.pkl", "rb") as f:
    best_tree = pickle.load(f)
with open("models/best_svc.pkl", "rb") as f:
    best_svc = pickle.load(f)
with open("models/best_rf.pkl", "rb") as f:
    best_rf = pickle.load(f)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    print(image_array.shape)
    image_array = image_array.reshape((1, -1))
    print(image_array.shape)
    result = {
        "Regressão Logística": reference_model.predict_proba(image_array).round(4).tolist(),
        "Regressão Logística balanceada": reference_model.predict_proba(image_array).round(4).tolist(),
        "Support Vector Machine": best_svc.predict_proba(image_array).round(4).tolist(),
        "Árvore de decisão": best_tree.predict_proba(image_array).round(4).tolist(),
        "Florestas aleatórias": best_rf.predict_proba(image_array).round(4).tolist(),
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

