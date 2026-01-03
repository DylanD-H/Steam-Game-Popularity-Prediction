import torch
import pandas as pd
from .preprocess import SteamPreprocessor
from .train import SteamNN
from fastapi import FastAPI
import uvicorn

device = "cuda" if torch.cuda.is_available() else "cpu"

preprocessor = SteamPreprocessor().load("./model/steam_preprocessor.pkl")
print(preprocessor.n_features)
model = SteamNN(preprocessor.n_features).to(device)
model.load_state_dict(torch.load("./model/steam_nn.pth", map_location=device))
model.eval()

threshold = 0.07

app = FastAPI(title="Steam Game Success Prediction")

@app.post("/predict")
def predict(game: dict):
    
    X_game, y_true = preprocessor.transform(pd.DataFrame([game]))
    X_game = torch.tensor(X_game, dtype=torch.float32)


    with torch.no_grad():
        logits = model(X_game.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        pred_class = (prob >= threshold)
    
    pred_class = "Successful" if pred_class else "Unsuccessful"
    act_class = "Successful" if y_true[0] == 1 else "Unsuccessful"

    return[{"key": "Game", "value": game['name']},
           {"key": "Predicted probability of success", "value": round(float(prob[0]),4)},
           {"key": "Predicted class", "value": pred_class},
           {"key": "Actual class", "value": act_class}
    ]

if __name__ == "__main__":
    uvicorn.run("predict:app", host='0.0.0.0', port=9696)