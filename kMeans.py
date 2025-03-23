import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OLD_DATA_PATH = "/Applications/webtemp/average-monthly-surface-temperature.csv"
PREDICTED_DATA_PATH = "/Applications/webtemp/temperature_forecast.csv"
KMEANS_RESULT_PATH = "/Applications/webtemp/kmeans_clusters.csv"
KMEANS_RESULT_PATH_BF_PREDICT = "/Applications/webtemp/kmeans_clusters_bf_predict.csv"



def apply_kmeans_before_predict(n_clusters=4):
    df = pd.read_csv(OLD_DATA_PATH)
    
    # 1. Tính nhiệt độ trung bình theo từng quốc gia
    df_grouped = df.groupby("Entity")["Average surface temperature.1"].mean().reset_index()

    # 2. Chạy K-Means trên dữ liệu tổng hợp
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_grouped["Kmeans_Cluster"] = kmeans.fit_predict(df_grouped[["Average surface temperature.1"]])

    # 3. Lưu kết quả phân cụm
    df_grouped.to_csv(KMEANS_RESULT_PATH_BF_PREDICT, index=False)
    print(f"K-Means phân cụm xong, kết quả lưu tại: {KMEANS_RESULT_PATH_BF_PREDICT}")

    return df_grouped

# Chạy K-Means
apply_kmeans_before_predict()




def apply_kmeans_clustering():
    if not os.path.exists(OLD_DATA_PATH) or not os.path.exists(PREDICTED_DATA_PATH):
        raise FileNotFoundError("Dữ liệu không đầy đủ")
    
    df_old = pd.read_csv(OLD_DATA_PATH)
    df_new = pd.read_csv(PREDICTED_DATA_PATH)

    # Xử lý dữ liệu
    df_old = df_old.groupby(["Entity", "year"])['Average surface temperature.1'].mean().reset_index()
    df_old.rename(columns={"Average surface temperature.1": "Average surface temperature"}, inplace=True)
    
    df_new = df_new.groupby(["Entity", "year"])['Average surface temperature'].mean().reset_index()
    
    df_combined = pd.concat([df_old, df_new])
    df_filtered = df_combined.pivot(index="Entity", columns="year", values="Average surface temperature")
    
    # Xử lý giá trị NaN
    df_filtered = df_filtered.interpolate(method="linear", axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
    df_filtered.dropna(axis=0, how="any", inplace=True)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered)

    
    # Áp dụng K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_filtered["Cluster"] = kmeans.fit_predict(X_scaled)
    df_filtered.reset_index(inplace=True)
    
    # Lưu kết quả
    df_filtered[["Entity", "Cluster"]].to_csv(KMEANS_RESULT_PATH, index=False)
    
    return {"message": f"Phân cụm bằng K-Means với K=4 thành công, dữ liệu đã được lưu."}


apply_kmeans_clustering()