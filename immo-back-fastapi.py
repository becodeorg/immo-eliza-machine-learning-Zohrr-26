
import os, json
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI

from eliza_new_id import New_ID

app = FastAPI()

#----------------------------------------------------------------

def load_model(path: str):
    booster = xgb.Booster()
    booster.load_model(path)
    return booster

#----------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model-o7.json")
booster = load_model(model_path); print("model loaded as booster")

## deploy with uvicorn

@app.get('/id-data')
def predict(input_data): # convert json into dict

    new_id = New_ID(**input_data)
    model_input = new_id.to_list()

    df_input = pd.DataFrame([model_input], columns=New_ID.fields)
    dmat = xgb.DMatrix(df_input)
    
    id_pred = booster.predict(dmat)[0]
    return id_pred

