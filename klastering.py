import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Baca data dari file CSV
df = pd.read_csv("penduduk.csv")

# Hitung rasio dan total
df["Jumlah Total"] = df["Jumlah Laki-laki"] + df["Jumlah Perempuan"]
df["Rasio"] = (df["Jumlah Laki-laki"] / df["Jumlah Perempuan"]) * 100

# Siapkan data untuk klastering
fitur = df[["Jumlah Laki-laki", "Jumlah Perempuan", "Rasio"]]
scaler = StandardScaler()
fitur_scaled = scaler.fit_transform(fitur)

# KMeans clustering (3 klaster)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(fitur_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Klastering 2D untuk visualisasi boundary
fitur_2d = df[["Jumlah Laki-laki", "Jumlah Perempuan"]].values
fitur_2d_scaled = scaler.fit_transform(fitur_2d)

kmeans_2d = KMeans(n_clusters=3, random_state=42)
kmeans_2d.fit(fitur_2d_scaled)
labels_2d = kmeans_2d.labels_
centroids_2d = scaler.inverse_transform(kmeans_2d.cluster_centers_)

# Meshgrid untuk boundary
x_min, x_max = fitur_2d[:, 0].min() - 1000, fitur_2d[:, 0].max() + 1000
y_min, y_max = fitur_2d[:, 1].min() - 1000, fitur_2d[:, 1].max() + 1000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = kmeans_2d.predict(grid_scaled)
Z = Z.reshape(xx.shape)

# Visualisasi klastering
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')

# Titik data
scatter = plt.scatter(df["Jumlah Laki-laki"], df["Jumlah Perempuan"],
                      c=labels_2d, cmap='viridis', s=100, edgecolors='k', label='Data Kota')

# Garis dan label kota
for i, row in df.iterrows():
    plt.plot([row["Jumlah Laki-laki"], row["Jumlah Laki-laki"] + 400],
             [row["Jumlah Perempuan"], row["Jumlah Perempuan"] + 400],
             color='gray', linestyle='--', linewidth=0.7)
    plt.text(row["Jumlah Laki-laki"] + 420, row["Jumlah Perempuan"] + 420,
             row["Kabupaten/Kota"], fontsize=9)

# Centroid
for idx, (x, y) in enumerate(centroids_2d):
    plt.scatter(x, y, marker='X', s=250, c='red', edgecolor='black', linewidth=1.5)
    plt.text(x + 300, y + 300, f'Centroid {idx}', fontsize=10, color='red', weight='bold')

# Legenda klaster
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Klaster 0', markerfacecolor=plt.cm.viridis(0.1), markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Klaster 1', markerfacecolor=plt.cm.viridis(0.5), markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Klaster 2', markerfacecolor=plt.cm.viridis(0.9), markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='X', color='w', label='Centroid', markerfacecolor='red', markeredgecolor='black', markersize=12)
]
plt.legend(handles=legend_elements, title="Keterangan", loc='upper left')

plt.xlabel("Jumlah Laki-laki")
plt.ylabel("Jumlah Perempuan")
plt.title("K-Means Klastering Penduduk dengan Centroid, Boundary & Label Kota")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# EVALUASI KLASTER: Silhouette Plot
# -------------------------

# Gunakan data 3D untuk evaluasi (Jumlah Laki-laki, Perempuan, Rasio)
silhouette_vals = silhouette_samples(fitur_scaled, df["Cluster"])
silhouette_avg = silhouette_score(fitur_scaled, df["Cluster"])

plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(3):  # jumlah klaster
    ith_cluster_silhouette_vals = silhouette_vals[df["Cluster"] == i]
    ith_cluster_silhouette_vals.sort()
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Rata-rata: {silhouette_avg:.2f}")
plt.xlabel("Nilai Silhouette")
plt.ylabel("Indeks Sampel")
plt.title("Evaluasi Klaster: Silhouette Plot")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# CETAK DAN SIMPAN HASIL KLASTER
# -------------------------

# Ambil nama kota dan klaster
kota_klaster = df[["Kabupaten/Kota", "Cluster"]].sort_values(by="Cluster")

# Cetak ke terminal
print("\n=== Daftar Kota/Kabupaten dan Klaster ===")
print(kota_klaster.to_string(index=False))

# Simpan ke file CSV
output_file = "hasil_klaster_kota.csv"
kota_klaster.to_csv(output_file, index=False)
print(f"\nHasil klaster telah disimpan ke file: {output_file}")