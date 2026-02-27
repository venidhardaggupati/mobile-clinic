"""
hour4_data_model.py — The Data Encoder (V2.0 Fleet Upgrade)
"""
import numpy as np
import pandas as pd
from typing import Any

def build_data_model(
    merged_df: pd.DataFrame,
    depot_index: int = 0,
    num_vehicles: int = 2,
) -> dict[str, Any]:
    """Builds the OR-Tools data dictionary from the merged village dataframe."""

    # 1. Location coordinates
    locations = list(zip(merged_df["Latitude"].astype(float), merged_df["Longitude"].astype(float)))
    n = len(locations)

    # 2. Active cases (integer array, NaN → 0)
    cases = merged_df["Active_Cases"].fillna(0).astype(int).tolist()

    # 3. Distance matrix (Haversine, metres)
    distance_matrix = _haversine_matrix(locations)

    # 4. Travel-time matrix (minutes, assuming avg 40 km/h)
    speed_mpm = 40_000 / 60          
    time_matrix = [
        [int(distance_matrix[i][j] / speed_mpm) for j in range(n)]
        for i in range(n)
    ]

    # 5. Village IDs (for labelling output)
    village_ids = merged_df["Village_ID"].astype(str).tolist()

    data = {
        "locations":        locations,
        "distance_matrix":  distance_matrix,
        "time_matrix":      time_matrix,
        "cases":            cases,          # NEW – dynamic service times
        "village_ids":      village_ids,
        "num_vehicles":     num_vehicles,   
        "depot":            depot_index,
    }

    return data


# ── Haversine helper ──────────────────────────────────────────────────────────
def _haversine_matrix(locations: list[tuple[float, float]]) -> list[list[int]]:
    """Return an N×N integer distance matrix in metres."""
    n = len(locations)
    lat = np.radians([loc[0] for loc in locations])
    lon = np.radians([loc[1] for loc in locations])
    R = 6_371_000  # Earth radius in metres

    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            dlat = lat[j] - lat[i]
            dlon = lon[j] - lon[i]
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat[i]) * np.cos(lat[j]) * np.sin(dlon / 2) ** 2
            )
            row.append(int(R * 2 * np.arcsin(np.sqrt(a))))
        matrix.append(row)
    return matrix