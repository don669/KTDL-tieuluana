from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = pd.read_csv("average-monthly-surface-temperature.csv")

@app.get("/get_top_countries")
def get_top_countries():
    avg_temp_by_country = df.groupby("Entity")["Average surface temperature"].mean().reset_index()

    avg_temp_by_country = avg_temp_by_country.sort_values(by="Average surface temperature")

    coldest_countries = avg_temp_by_country.head(5).to_dict(orient="records")
    hottest_countries = avg_temp_by_country.tail(5).to_dict(orient="records")

    return {"coldest": coldest_countries, "hottest": hottest_countries}


def apply_dbscan():
    df_filtered = df[["Entity", "year", "Average surface temperature"]].copy()
    df_filtered = df_filtered.groupby(["Entity", "year"], as_index=False)["Average surface temperature"].mean()


    pivot_df = df_filtered.pivot(index="Entity", columns="year", values="Average surface temperature")

    pivot_df = pivot_df.interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot_df)

    dbscan = DBSCAN(eps=1.2, min_samples=3)  
    clusters = dbscan.fit_predict(X_scaled)

    pivot_df["Cluster"] = clusters
    pivot_df.reset_index(inplace=True)

    return pivot_df[["Entity", "Cluster"]].to_dict(orient="records")



@app.get("/get_temperature")
def get_temperature(country: str):
    country_data = df[df["Entity"] == country]

    avg_temp_by_year = country_data.groupby("year")["Average surface temperature"].mean().reset_index()

    return avg_temp_by_year.to_dict(orient="records")

@app.get("/get_clusters")
def get_clusters():
    return apply_dbscan()
