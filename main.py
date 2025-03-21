from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
from randomforest import apply_dbscan_predict1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dữ liệu
DATA_PATH = "average-monthly-surface-temperature.csv"
PREDICTED_DATA_PATH = "temperature_forecast.csv"

df = pd.read_csv(DATA_PATH)

@app.get("/get_top_countries")
def get_top_countries():
    avg_temp_by_country = df.groupby("Entity")["Average surface temperature"].mean().reset_index()
    avg_temp_by_country = avg_temp_by_country.sort_values(by="Average surface temperature")
    return {
        "coldest": avg_temp_by_country.head(5).to_dict(orient="records"),
        "hottest": avg_temp_by_country.tail(5).to_dict(orient="records"),
    }

def apply_dbscan():
    # Xử lý dữ liệu trùng lặp trước khi pivot
    df_filtered = df.groupby(["Entity", "year"])["Average surface temperature"].mean().reset_index()

    # Pivot dữ liệu (bây giờ không còn trùng lặp)
    df_pivot = df_filtered.pivot(index="Entity", columns="year", values="Average surface temperature")

    # Xử lý NaN
    df_pivot = df_pivot.interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
    df_pivot.dropna(axis=0, how="any", inplace=True)

    # Scale dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pivot)

    # Áp dụng DBSCAN
    dbscan = DBSCAN(eps=1.2, min_samples=3)
    clusters = dbscan.fit_predict(X_scaled)

    # Gán cluster vào DataFrame
    df_pivot["Cluster"] = clusters
    df_pivot.reset_index(inplace=True)

    return df_pivot[["Entity", "Cluster"]].to_dict(orient="records")

@app.get("/get_temperature")
def get_temperature(country: str):
    country_data = df[df["Entity"] == country]
    avg_temp_by_year = country_data.groupby("year")["Average surface temperature"].mean().reset_index()
    return avg_temp_by_year.to_dict(orient="records")

@app.get("/get_clusters")
def get_clusters():
    return apply_dbscan()

@app.get("/predict_temperature")
def predict_temperature(entity: str, years: int = 10):
    country_data = df[df["Entity"] == entity]
    if country_data.empty:
        raise HTTPException(status_code=404, detail="Quốc gia không tồn tại trong dữ liệu")
    
    avg_temp_by_year = country_data.groupby("year")["Average surface temperature"].mean().reset_index()
    avg_temp_by_year.set_index("year", inplace=True)

    # Giới hạn years tối đa để tránh overfitting
    years = min(years, 20)

    model = sm.tsa.statespace.SARIMAX(avg_temp_by_year, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    future_years = list(range(int(avg_temp_by_year.index[-1]) + 1, int(avg_temp_by_year.index[-1]) + years + 1))
    forecast = results.forecast(steps=years)
    
    return {
        "entity": entity,
        "forecast": {int(year): float(temp) for year, temp in zip(future_years, forecast.tolist())}
    }

def apply_dbscan_predict():
    # Kiểm tra file có tồn tại không
    if not os.path.exists(PREDICTED_DATA_PATH):
        raise HTTPException(status_code=404, detail="Dữ liệu dự báo không tồn tại")

    df_pred = pd.read_csv(PREDICTED_DATA_PATH)
    df_filtered = df_pred.pivot(index="Entity", columns="year", values="Average surface temperature")
    
    df_filtered = df_filtered.interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
    df_filtered.dropna(axis=0, how="any", inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered)

    dbscan = DBSCAN(eps=1.2, min_samples=3)
    clusters = dbscan.fit_predict(X_scaled)

    df_filtered["Cluster"] = clusters
    df_filtered.reset_index(inplace=True)

    return df_filtered[["Entity", "Cluster"]].to_dict(orient="records")

@app.get("/get_clusters_predict")
def get_clusters_predict():
    return apply_dbscan_predict1()
