import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from fastapi import HTTPException

OLD_DATA_PATH = "/Applications/webtemp/average-monthly-surface-temperature.csv"  # Dữ liệu 1940-2024
PREDICTED_DATA_PATH = "/Applications/webtemp/temperature_forecast.csv"  # Dữ liệu 2025-2044

def apply_dbscan_predict1():
    if not os.path.exists(OLD_DATA_PATH) or not os.path.exists(PREDICTED_DATA_PATH):
        raise HTTPException(status_code=404, detail="Dữ liệu không đầy đủ")

    df_old = pd.read_csv(OLD_DATA_PATH)
    df_new = pd.read_csv(PREDICTED_DATA_PATH)

    # Xử lý dữ liệu cũ: Trung bình nhiệt độ để tránh trùng lặp
    df_old = df_old.groupby(["Entity", "year"])["Average surface temperature.1"].mean().reset_index()
    df_old.rename(columns={"Average surface temperature.1": "Average surface temperature"}, inplace=True)

    # Xử lý dữ liệu mới: Nếu có trùng lặp, chọn trung bình
    df_new = df_new.groupby(["Entity", "year"])["Average surface temperature"].mean().reset_index()

    # Gộp dữ liệu
    df_combined = pd.concat([df_old, df_new])

    # Pivot lại dữ liệu sau khi xử lý trùng lặp
    df_filtered = df_combined.pivot(index="Entity", columns="year", values="Average surface temperature")

    # Xử lý NaN
    df_filtered = df_filtered.interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
    df_filtered.dropna(axis=0, how="any", inplace=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered)

    # Áp dụng DBSCAN
    dbscan = DBSCAN(eps=1.2, min_samples=3)
    clusters = dbscan.fit_predict(X_scaled)

    # Gán nhãn cụm
    df_filtered["Cluster"] = clusters
    df_filtered.reset_index(inplace=True)

    return df_filtered[["Entity", "Cluster"]].to_dict(orient="records")
