import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import xgboost as xgb

#-----------------------------------------
#----------------- SETUP -----------------
#-----------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, "data", "Kangaroo.csv")

df = pd.read_csv(csv_file) # index_col="id"

pd.set_option('future.no_silent_downcasting', True)

df['id'] = df['id'].astype(int)
df = df.drop_duplicates(subset='id', keep='first')
df = df.set_index('id')

#-----------------------------------------
#-------------- Black Magic --------------
#-----------------------------------------

giraffe_file = os.path.join(current_dir, "data", "Giraffe.csv")
giraffe = pd.read_csv(giraffe_file)
giraffe = giraffe.rename(columns={"propertyId": "id"})

coords = giraffe[[
        "id", "longitude", "latitude", # 44.513
        "cadastralIncome", # 44.142
        "primaryEnergyConsumptionPerSqm", # 43.941
        
        ]]

df = df.merge(coords, on="id", how="left")

df = df.dropna(subset=['longitude'])

#-----------------------------------------
#--------- AI Model Province -------------
#-----------------------------------------

print('')
print(f"rows: {df.shape[0]}")
print(f"columns: {df.shape[1]}")
print('')

province = {
    'Brussels': 5,

    'Antwerp': 11,
    'Flemish Brabant': 12,
    'East Flanders': 13,
    'West Flanders': 14,
    'Limburg': 15,

    'Luxembourg': 21,
    'Liège': 22,
    'Walloon Brabant': 23,
    'Namur': 24,
    'Hainaut': 25,
}

df['province'] = df["province"].replace(province)

#-----------------------------------------
#----------- Columns Drop ----------------
#-----------------------------------------

del_cols = ['Unnamed: 0', 'url', 'hasBalcony', 'accessibleDisabledPeople', 'monthlyCost',] # empty
df = df.drop(del_cols, axis=1)

#-----------------------------------------
#----------- Drop Trash Rows -------------
#-----------------------------------------

df = df.dropna(subset=['bedroomCount'])
df = df.dropna(subset=['habitableSurface'])

#df = df.dropna(subset=['bathroomCount']); print("dropna bathroomCount")
#df = df.dropna(subset=['toiletCount']); print("dropna toiletCount")

# df = df[df['cadastralIncome'] == -1]

# print(df.columns); print('')

#-----------------------------------------
#--------------- AI Model ----------------
#-------------- str to int ---------------
#-----------------------------------------

str_cols = [
    'locality',
    'terraceOrientation', 
    'epcScore', 
    'gardenOrientation', 
    'kitchenType',
    'heatingType',
    'floodZoneType',
    'buildingCondition',
    'type',
    'subtype',
    ]

def col_str_to_int(df, col):
    numbers, _ = pd.factorize(df[col])
    return numbers + 1

for col in str_cols:
    df[col] = col_str_to_int(df, col) 

#-----------------------------------------
#--------- % for price range -------------
#-----------------------------------------

df = df.dropna(subset=['price'])

print(f"numbers of properties in the datasets: {df.shape[0]}")
print('')

count = df[df['price'] < 1_000_000].shape[0]
cent = int(count / df.shape[0] * 100)
mean = df[df['price'] < 1_000_000]['price'].mean()
print(f"% of properties below 1.000.000 -- {count} -- {cent}% -- {mean:,.0f}")

count = df[df['price'] < 600_000].shape[0]
cent = int(count / df.shape[0] * 100)
mean = int(df[df['price'] < 600_000]['price'].mean())
print(f"% of properties below 600.000 ---- {count} -- {cent}% -- {mean:,.0f}")

count = df[df['price'] < 500_000].shape[0]
cent = int(count / df.shape[0] * 100)
mean = int(df[df['price'] < 500_000]['price'].mean())
print(f"% of properties below 500.000 ---- {count} -- {cent}% -- {mean:,.0f}")

#-----------------------------------------
#-------- Drop Empty Price Row -----------
#-----------------------------------------

df = df.dropna(subset=['price'])

print('')
df = df[df['price'] < 1_000_000]; print("94%: only below 1.000.000")
# df = df[df['price'] < 600_000]; print("84%: only below 600.000")

#-----------------------------------------
#--------- bool & object cols ------------
#-----------------------------------------

df = df.replace({True: 1, False: 0})
df = df.fillna(-1)
#-----------------------------------------

obj_cols = df.select_dtypes(include='object').columns
df[obj_cols] = df[obj_cols].astype(int)

#-----------------------------------------
#----------- create new csv --------------
#-----------------------------------------

pandas_csv = os.path.join(current_dir, "data", "model.csv")
df.to_csv(pandas_csv, index=True, sep=',', encoding="utf-8")

#-----------------------------------------
#----------- Split Dataset ---------------
#-----------------------------------------

X = df.drop(columns=['price'])
y = df['price']

immo_train, immo_test, price_train, price_test = train_test_split(X, y, test_size=0.05, random_state=42)

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

#-----------------------------------------
#------- Cross-Validation Evaluation -----
#-----------------------------------------

print('')

cv_scores = -cross_val_score(
    model, immo_train, price_train,
    
    scoring='neg_mean_absolute_error',
    cv=3, 
    n_jobs=-1
)
print(f"3-fold CV MAE: {cv_scores.mean():,.2f} ± {cv_scores.std():.2f}") 
   

model.fit(immo_train, price_train)

""" model.set_params(n_estimators=2_000)
model.fit(immo_train, price_train, xgb_model=model.get_booster())

model.set_params(n_estimators=3_000)
model.fit(immo_train, price_train, xgb_model=model.get_booster()) """

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

train_pred = model.predict(immo_train)

mae = mean_absolute_error(price_train, train_pred)
accuracy = 100 - mean_absolute_percentage_error(price_train, train_pred) * 100

print('')
print(f"Train: Accuracy: {accuracy:.2f}%     MAE: {mae:,.2f}")

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

price_pred = model.predict(immo_test)

mae = mean_absolute_error(price_test, price_pred)
accuracy = 100 - mean_absolute_percentage_error(price_test, price_pred) * 100

residuals = price_test - price_pred

""" plt.figure(figsize=(12,8))
plt.hist(residuals, bins=80, edgecolor='black')
plt.axvline(0, color='k', linestyle='--')
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend([f"MAE: {mae:,.2f}"])
plt.title(f"Histogram of Residuals: {len(residuals)}")
plt.xlabel("Error between price and prediction")
plt.ylabel("Number of properties")
plt.show()  """

""" plt.figure(figsize=(6,4))
plt.scatter(price_pred, residuals, alpha=0.4)
plt.axhline(50_000, color='orange', linestyle='--', label='50k')
plt.axhline(100_000, color='red',   linestyle='--', label='100k')
plt.axhline(-50_000, color='orange', linestyle='--')
plt.axhline(-100_000, color='red',   linestyle='--')
plt.title("Error and property value")
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xlabel("Property Value")
plt.ylabel("Error")
plt.show()  """

#-----------------------------------------
#--------------- Metrics -----------------
#-----------------------------------------

print('')
print(f"Test: Accuracy: {accuracy:.2f}%      MAE: {mae:,.2f}")
print('')



""" 
feat_imp = pd.Series(model.feature_importances_, index=immo_train.columns).sort_values(ascending=False)
for feat, imp in feat_imp.items(): print(f"{feat:25s} → {imp*100:.2f}")
"""
