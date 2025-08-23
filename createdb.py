import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
import json # <--- 1. IMPORT THE JSON LIBRARY

# -------------------------------
# 1. Create Dummy SQL Database (No changes here)
# -------------------------------
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE full_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    tag TEXT CHECK(tag IN ('unverified', 'duplicate', 'unique')) NOT NULL,
    category TEXT CHECK(category IN ('pothole','trash')) NOT NULL
);
""")
cursor.execute("CREATE INDEX idx_tag ON full_reports(tag);")
dummy_data = [
    # Pothole cluster 1
    (12.9716, 77.5946, "unverified", "pothole"),
    (12.9717, 77.5947, "unverified", "pothole"),
    (12.9718, 77.5948, "unverified", "pothole"),

    # Pothole cluster 2
    (13.0010, 77.6000, "unverified", "pothole"),
    (13.0011, 77.6001, "unverified", "pothole"),
    (13.0012, 77.6002, "unverified", "pothole"),

    # Trash cluster 1
    (12.9352, 77.6245, "unverified", "trash"),
    (12.9353, 77.6246, "unverified", "trash"),
    (12.9354, 77.6247, "unverified", "trash"),

    # Trash cluster 2
    (12.9500, 77.6300, "unverified", "trash"),
    (12.9501, 77.6301, "unverified", "trash"),
    (12.9502, 77.6302, "unverified", "trash"),

    # Some duplicates and unique (ignored)
    (12.9600, 77.6400, "duplicate", "trash"),
    (12.9650, 77.6450, "unique", "pothole"),
]
cursor.executemany("INSERT INTO full_reports(latitude,longitude,tag,category) VALUES (?,?,?,?)", dummy_data)
conn.commit()

# -------------------------------
# 2. Query unverified reports (No changes here)
# -------------------------------
cursor.execute("SELECT id, latitude, longitude, category FROM full_reports WHERE tag='unverified'")
rows = cursor.fetchall()
pothole, pothole_ids, trash, trash_ids = [], [], [], []
for r in rows:
    _id, lat, lon, category = r
    if category == "pothole":
        pothole.append([lat, lon])
        pothole_ids.append(_id)
    elif category == "trash":
        trash.append([lat, lon])
        trash_ids.append(_id)

# -------------------------------
# 3. Run DBSCAN with Offset (No changes here)
# -------------------------------
def cluster_and_map(data, ids, eps=0.001, min_samples=2, cluster_id_offset=0):
    if not data:
        return {}, cluster_id_offset
    X = np.array(data)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    mapped_clusters = {
        ids[i]: int(labels[i] + cluster_id_offset) if labels[i] != -1 else -1
        for i in range(len(ids))
    }
    num_new_clusters = np.max(labels) + 1
    return mapped_clusters, cluster_id_offset + num_new_clusters

next_cluster_id = 0
pothole_clusters, next_cluster_id = cluster_and_map(pothole, pothole_ids, cluster_id_offset=next_cluster_id)
trash_clusters, next_cluster_id = cluster_and_map(trash, trash_ids, cluster_id_offset=next_cluster_id)

# -------------------------------
# 4. Output as JSON
# -------------------------------
# --- 2. COMBINE RESULTS INTO A SINGLE DICTIONARY ---
final_output = {
    "pothole_clusters": pothole_clusters,
    "trash_clusters": trash_clusters
}

# --- 3. PRINT THE DICTIONARY AS A JSON STRING ---
print(json.dumps(final_output, indent=4))