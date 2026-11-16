"""Coverage metrics on the native HYCOM horizontal grid."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


D2KM = 111.0
HYCOM_LAT_DEG = 0.04
HYCOM_LON_DEG = 0.08


def native_hycom_cells(
    lats: Iterable[float],
    lons: Iterable[float],
    ocean_valid: Iterable[bool] | None = None,
) -> set[tuple[int, int]]:
    """Return unique native HYCOM cells visited by valid ocean positions."""
    lat_values = np.asarray(list(lats), dtype=np.float64).ravel()
    lon_values = np.asarray(list(lons), dtype=np.float64).ravel()
    if ocean_valid is None:
        valid = np.ones(lat_values.shape, dtype=bool)
    else:
        valid = np.asarray(list(ocean_valid), dtype=bool).ravel()
    return {
        (
            int(np.floor(lat / HYCOM_LAT_DEG)),
            int(np.floor(lon / HYCOM_LON_DEG)),
        )
        for lat, lon in zip(lat_values[valid], lon_values[valid])
    }


def native_hycom_coverage_area(cells: set[tuple[int, int]]) -> float:
    """Sum unique native-grid cell areas, including latitude dependence."""
    area = 0.0
    for i, _j in cells:
        cell_lat = (i + 0.5) * HYCOM_LAT_DEG
        lat_width_km = HYCOM_LAT_DEG * D2KM
        lon_width_km = (
            HYCOM_LON_DEG
            * D2KM
            * np.cos(np.deg2rad(cell_lat))
        )
        area += float(lat_width_km * lon_width_km)
    return area


def native_hycom_weighted_coverage(
    lats: Iterable[float],
    lons: Iterable[float],
    speeds: Iterable[float],
    ocean_valid: Iterable[bool] | None = None,
) -> float:
    """Integrate representative current speed over unique native HYCOM cells."""
    lat_values = np.asarray(list(lats), dtype=np.float64).ravel()
    lon_values = np.asarray(list(lons), dtype=np.float64).ravel()
    speed_values = np.asarray(list(speeds), dtype=np.float64).ravel()
    if not (
        lat_values.size == lon_values.size == speed_values.size
    ):
        raise ValueError("lat, lon, and speed arrays must have equal lengths")
    if ocean_valid is None:
        valid = np.ones(lat_values.shape, dtype=bool)
    else:
        valid = np.asarray(list(ocean_valid), dtype=bool).ravel()
    if valid.size != lat_values.size:
        raise ValueError("ocean_valid must match the position arrays")

    cell_sums: dict[tuple[int, int], float] = {}
    cell_counts: dict[tuple[int, int], int] = {}
    for lat, lon, speed in zip(
        lat_values[valid],
        lon_values[valid],
        speed_values[valid],
    ):
        cell = (
            int(np.floor(lat / HYCOM_LAT_DEG)),
            int(np.floor(lon / HYCOM_LON_DEG)),
        )
        cell_sums[cell] = cell_sums.get(cell, 0.0) + float(speed)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

    value = 0.0
    for cell, speed_sum in cell_sums.items():
        cell_speed = speed_sum / cell_counts[cell]
        value += native_hycom_coverage_area({cell}) * cell_speed
    return float(value)


def native_cell_center(
    cell: tuple[int, int],
) -> tuple[float, float]:
    i, j = cell
    return (
        (i + 0.5) * HYCOM_LAT_DEG,
        (j + 0.5) * HYCOM_LON_DEG,
    )
