

import numpy as np
import pandas as pd
import os

import xgboost as xgb
import streamlit as st

#-----------------------------------------
#----------------- SETUP -----------------
#-----------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "data", "model.csv")

df = pd.read_csv(csv_file, index_col="id")

print(df.head(2))

X = df.drop(columns=['price'])
y = df['price']

#-----------------------------------------
#--------------- XG Boost ---------------- 
#------- Extreme Gradient Boosting -------
#-----------------------------------------

model = xgb.XGBRegressor(
    n_estimators=8000,
    learning_rate=0.022, #0.031
    max_depth=9, # 9
    subsample=0.78764, #0.831
    colsample_bytree=0.432237, # 0.395
    colsample_bylevel=0.994548,
    reg_alpha=0.83555,
    reg_lambda=0.3949,
    random_state=42,
    tree_method='hist'
    )

model.fit(X, y)

#-----------------------------------------
#------------ Save Model -----------------
#-----------------------------------------

o7 = os.path.join(current_dir, "model-o7.json")
model.save_model(o7)
