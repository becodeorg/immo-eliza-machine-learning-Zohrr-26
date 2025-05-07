import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import xgboost as xgb

#-----------------------------------------
#----------------- SETUP -----------------
#-----------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "data", "clean_csv.csv")
dirty_csv = os.path.join(current_dir, "data", "dirty_csv.csv")

df = pd.read_csv(csv_file, index_col="id")
dirty_df = pd.read_csv(dirty_csv, index_col="id")

pd.set_option('future.no_silent_downcasting', True)

#-----------------------------------------
#--------- % for price range -------------
#-----------------------------------------

df = df.dropna(subset=['price'])
df = df.drop(columns=['error_sum'], errors='ignore')
dirty_df = dirty_df.drop(columns=['error_sum'], errors='ignore')

print(f"numbers of properties in the datasets: {df.shape[0]}")
print('')

count = df[df['price'] < 1_000_000].shape[0]
print(f"% of properties below 1.000.000 -- {count} -- {int(count / df.shape[0] * 100)}%")

count = df[df['price'] < 600_000].shape[0]
print(f"% of properties below 600.000 ---- {count} -- {int(count / df.shape[0] * 100)}%")

count = df[df['price'] < 500_000].shape[0]
print(f"% of properties below 500.000 ---- {count} -- {int(count / df.shape[0] * 100)}%")

#-----------------------------------------
#-------- Drop Empty Price Row -----------
#-----------------------------------------

df = df.dropna(subset=['price'])

print('')
df = df[df['price'] < 1_000_000]; print("only below 1.000.000")
# df = df[df['price'] < 500_000]; print("only below 500.000")

#-----------------------------------------
#----------- Random Forest ---------------
#-----------------------------------------

z = df.drop(columns=['price'])
y = df['price']

immo_train, immo_test, price_train, price_test = train_test_split(z, y, test_size=0.25, random_state=42)


z = dirty_df.drop(columns=["price"])
y = dirty_df['price']

trash_train, trash_test, price_t_train, price_t_test = train_test_split(z, y, test_size=0.1, random_state=42)

#-----------------------------------------
#--------------- XG Boost ---------------- 
#------- Extreme Gradient Boosting -------
#-----------------------------------------

model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.031,
    max_depth=9, # 9
    subsample=0.831, #0.831
    colsample_bytree=0.395, # 0.395
    colsample_bylevel=0.449,
    reg_alpha=1.769,
    reg_lambda=3.922,
    random_state=42,
    tree_method='hist', 
    )

model.fit(trash_train, price_t_train)

model.set_params(n_estimators=2000)  
model.fit(immo_train, price_train, xgb_model=model.get_booster())

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

train_pred = model.predict(immo_train)

mae = mean_absolute_error(price_train, train_pred)
accuracy = 100 - mean_absolute_percentage_error(price_train, train_pred) * 100

print('')
print(f"Train: Accuracy: {accuracy:.2f}%")
print(f"Train:     MAE: {mae:,.2f}")
print('')

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

dirty_pred = model.predict(trash_test)

mae = mean_absolute_error(price_t_test, dirty_pred)
accuracy = 100 - mean_absolute_percentage_error(price_t_test, dirty_pred) * 100

print('')
print(f"Trash: Accuracy: {accuracy:.2f}%")
print(f"Trash:     MAE: {mae:,.2f}")
print('')

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

price_pred = model.predict(immo_test)

mae = mean_absolute_error(price_test, price_pred)
accuracy = 100 - mean_absolute_percentage_error(price_test, price_pred) * 100

print('')
print(f"Test: Accuracy: {accuracy:.2f}%")
print(f"Test:     MAE: {mae:,.2f}")
print('')

""" feat_imp = pd.Series(model.feature_importances_, index=immo_train.columns).sort_values(ascending=False)
for feat, imp in feat_imp.items(): print(f"{feat:25s} → {imp*100:.2f}") """

