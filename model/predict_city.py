import pickle, pandas as pd, numpy as np
from pathlib import Path
import xgboost as xgb

ROOT = Path('D:/ISROPS1/data-pipeline')

CORE_FEATURES = ['ndvi','ndwi','ndbi','albedo','tatm','era5_humidity','era5_wind_speed','doy_sin']

# Load Kolkata data, train on core features only
df = pd.read_parquet(ROOT/'outputs/exports/kolkata/feature_matrix_kolkata_ALL.parquet')
df = df[df['lst_celsius'] > 10].dropna(subset=CORE_FEATURES + ['lst_celsius'])

X = df[CORE_FEATURES].values
y = df['lst_celsius'].values

model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
model.fit(X, y)

# Test on Delhi
df_d = pd.read_parquet(ROOT/'outputs/exports/delhi/feature_matrix_delhi_ALL.parquet')
df_d = df_d[df_d['lst_celsius'] > 10].dropna(subset=CORE_FEATURES)
pred = model.predict(df_d[CORE_FEATURES].values)

print(f'Delhi pixels: {len(df_d):,}')
print(f'Predicted LST: mean={pred.mean():.2f}C  min={pred.min():.2f}C  max={pred.max():.2f}C')
print(f'Actual LST:    mean={df_d["lst_celsius"].mean():.2f}C')

# Save this city-agnostic model
pickle.dump({'model': model, 'features': CORE_FEATURES}, open(ROOT/'model/outputs/xgb_city_model.pkl','wb'))
print('City model saved: xgb_city_model.pkl')