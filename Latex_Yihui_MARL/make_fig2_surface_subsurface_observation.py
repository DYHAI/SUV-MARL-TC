"""Recreate the paired surface-wind/subsurface-current overview figure.

The layout deliberately uses a dedicated colorbar row so that the colorbar
labels cannot overlap the figure footnotes or the neighboring panel.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from surface_wind_env import land_ocean_mask


STORM = "HINNAMNOR_2022"
CENTER_TIME = pd.Timestamp("2022-09-03 21:00:00")
WINDOW = (118.5, 131.0, 21.0, 31.0)
AUV_IMAGE = np.fliplr(plt.imread(HERE / "figures" / "auv_icon.png"))


def subset(lat, lon, *fields):
    lon0, lon1, lat0, lat1 = WINDOW
    j = (lon >= lon0) & (lon <= lon1)
    i = (lat >= lat0) & (lat <= lat1)
    cropped = [a[i][:, j] for a in fields]
    return lon[j], lat[i], cropped


def draw_land(ax, ocean, lon, lat):
    ax.contourf(lon, lat, ocean, levels=[-0.1, 0.5],
                colors=["#d9d2c3"], zorder=0)
    ax.contour(lon, lat, ocean, levels=[0.5], colors="#5b5b5b",
               linewidths=0.9, zorder=4)


def draw_track(ax):
    tr = pd.read_csv(
        ROOT / "data" / "selected_typhoons" / STORM / "track.csv",
        usecols=["ISO_TIME", "LAT", "LON"],
        low_memory=False,
    )
    tr["ISO_TIME"] = pd.to_datetime(tr["ISO_TIME"], errors="coerce")
    tr["LAT"] = pd.to_numeric(tr["LAT"], errors="coerce")
    tr["LON"] = pd.to_numeric(tr["LON"], errors="coerce")
    tr = tr.dropna().sort_values("ISO_TIME")
    lon0, lon1, lat0, lat1 = WINDOW
    view = tr[tr.LON.between(lon0, lon1) & tr.LAT.between(lat0, lat1)]
    ax.plot(view.LON, view.LAT, color="black", lw=1.8, ls=(0, (4, 4)),
            zorder=6)

    i = (view.ISO_TIME - CENTER_TIME).abs().idxmin()
    j = min(int(view.index.get_loc(i)) + 3, len(view) - 1)
    x0, y0 = view.iloc[j][["LON", "LAT"]]
    if j + 2 < len(view):
        x1, y1 = view.iloc[j + 2][["LON", "LAT"]]
    else:
        x1, y1 = x0, y0 + 0.5
    ax.annotate(
        "IBTrACS track",
        xy=(x1, y1),
        xytext=(x1 + 0.45, y1 + 0.15),
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.92),
        zorder=9,
    )
    return view


def draw_eye(ax, lon, lat):
    ax.scatter([lon], [lat], s=130, facecolor="white", edgecolor="black",
               linewidth=2.0, zorder=10)
    ax.scatter([lon], [lat], s=35, color="black", zorder=11)
    ax.annotate("Eye", xy=(lon, lat), xytext=(lon, lat - 0.85),
                fontsize=11, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                          alpha=0.78),
                arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8),
                zorder=12)


def draw_auvs(ax, center_lon, center_lat):
    radius_km = 150.0
    angles = np.deg2rad([90, 0, 270, 180])
    for k, ang in enumerate(angles, start=1):
        x = center_lon + radius_km * np.cos(ang) / (
            111.0 * np.cos(np.deg2rad(center_lat))
        )
        y = center_lat + radius_km * np.sin(ang) / 111.0
        image = OffsetImage(AUV_IMAGE, zoom=0.18)
        ab = AnnotationBbox(
            image,
            (x, y),
            frameon=False,
            pad=0.0,
            box_alignment=(0.5, 0.5),
            zorder=13,
        )
        ax.add_artist(ab)
        ha = "center"
        va = "center"
        xt, yt = x, y
        if ang == np.deg2rad(90):
            xt, yt, va = x, y + 0.38, "bottom"
        elif ang == np.deg2rad(270):
            xt, yt, va = x, y - 0.38, "top"
        elif ang == 0:
            xt, yt, ha = x + 0.55, y, "left"
        else:
            xt, yt, ha = x - 0.55, y, "right"
        ax.annotate(
            f"SUV {k}",
            xy=(x, y),
            xytext=(xt, yt),
            ha=ha,
            va=va,
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#7d95b2",
                      lw=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#2c5385", lw=0.7),
            zorder=14,
        )


def add_horizontal_colorbar(fig, cax, mappable, label):
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=11, length=4, pad=5)
    cb.set_label(label, fontsize=13, labelpad=8)
    cb.outline.set_linewidth(0.8)


def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.linewidth": 0.9,
    })

    wet = xr.open_dataset(
        ROOT / "data" / "era5_wind" / STORM / "era5_10m_wind_hourly.nc"
    ).sel(time=CENTER_TIME, method="nearest")
    u = wet["u10"].values.astype(float)
    v = wet["v10"].values.astype(float)
    wlat = wet["latitude"].values.astype(float)[::-1]
    wlon = wet["longitude"].values.astype(float)
    u = u[::-1]
    v = v[::-1]
    wet.close()

    cur = xr.open_dataset(
        ROOT / "data" / "union_region" / STORM / "hycom_uv_30m.nc"
    ).sel(time=CENTER_TIME, method="nearest").isel(depth=0)
    cu = cur["water_u"].values.astype(float)
    cv = cur["water_v"].values.astype(float)
    clat = cur["lat"].values.astype(float)
    clon = cur["lon"].values.astype(float)
    cur.close()

    ocean, _, olon, olat = land_ocean_mask()
    wlon, wlat, (u, v, ocean_w) = subset(wlat, wlon, u, v, ocean)
    clon, clat, (cu, cv) = subset(clat, clon, cu, cv)
    ocean_c = np.isfinite(cu).astype(float)

    wspd = np.hypot(u, v)
    cspd = np.hypot(cu, cv)
    wspd = np.where(ocean_w >= 0.5, wspd, np.nan)
    cspd = np.where(ocean_c >= 0.5, cspd, np.nan)

    track = pd.read_csv(
        ROOT / "data" / "selected_typhoons" / STORM / "track.csv",
        usecols=["ISO_TIME", "LAT", "LON"],
        low_memory=False,
    )
    track["ISO_TIME"] = pd.to_datetime(track["ISO_TIME"], errors="coerce")
    track["LAT"] = pd.to_numeric(track["LAT"], errors="coerce")
    track["LON"] = pd.to_numeric(track["LON"], errors="coerce")
    track = track.dropna().sort_values("ISO_TIME")
    eye = track.iloc[(track.ISO_TIME - CENTER_TIME).abs().argmin()]
    center_lon, center_lat = float(eye.LON), float(eye.LAT)

    fig = plt.figure(figsize=(16.0, 8.2), dpi=300)
    gs = GridSpec(
        2, 2, figure=fig, height_ratios=[10.0, 0.62],
        left=0.065, right=0.975, top=0.865, bottom=0.095,
        wspace=0.20, hspace=0.34,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax1 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])
    fig.suptitle("Multi-SUV surface-subsurface observation under TC conditions",
                 fontsize=24, fontweight="bold", y=0.965)

    draw_land(ax1, ocean_w, wlon, wlat)
    im1 = ax1.pcolormesh(wlon, wlat, wspd, shading="auto",
                         cmap="YlGnBu_r", vmin=0, vmax=30, zorder=1)
    ax1.streamplot(wlon, wlat, u, v, color="white", density=1.25,
                   linewidth=0.65, arrowsize=0.6, zorder=2)
    draw_track(ax1)
    draw_eye(ax1, center_lon, center_lat)
    draw_auvs(ax1, center_lon, center_lat)
    ax1.set_xlim(WINDOW[:2])
    ax1.set_ylim(WINDOW[2:])
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.text(0.0, 1.035, "(a) Surface wind field (ERA5, 10 m)",
             transform=ax1.transAxes, ha="left", va="bottom",
             fontsize=17, fontweight="bold")
    ax1.grid(color="white", alpha=0.18, lw=0.5)
    add_horizontal_colorbar(fig, cax1, im1, "Wind speed (m s$^{-1}$)")

    draw_land(ax2, ocean_c, clon, clat)
    im2 = ax2.pcolormesh(clon, clat, cspd, shading="auto",
                         cmap="YlGn", vmin=0, vmax=2.0, zorder=1)
    ax2.contour(clon, clat, cspd, levels=np.arange(0.2, 2.01, 0.2),
                colors="0.35", linewidths=0.45, alpha=0.65, zorder=2)
    target_lon = np.linspace(clon[0], clon[-1], 120)
    target_lat = np.linspace(clat[0], clat[-1], 84)
    gx, gy = np.meshgrid(target_lon, target_lat)
    points = np.column_stack([gy.ravel(), gx.ravel()])
    iu = RegularGridInterpolator(
        (clat, clon), np.nan_to_num(cu), bounds_error=False, fill_value=0.0
    )(points).reshape(gx.shape)
    iv = RegularGridInterpolator(
        (clat, clon), np.nan_to_num(cv), bounds_error=False, fill_value=0.0
    )(points).reshape(gx.shape)
    ax2.streamplot(target_lon, target_lat, iu, iv, color="0.30",
                   density=0.80, linewidth=0.55, arrowsize=0.55, zorder=3)
    draw_track(ax2)
    draw_eye(ax2, center_lon, center_lat)
    draw_auvs(ax2, center_lon, center_lat)

    yy, xx = np.meshgrid(clat, clon, indexing="ij")
    dist = np.hypot(
        (yy - center_lat) * 111.0,
        (xx - center_lon) * 111.0 * np.cos(np.deg2rad(center_lat)),
    )
    local = np.where(dist <= 180.0, np.nan_to_num(cspd, nan=-1.0), -1.0)
    imax = np.unravel_index(np.nanargmax(local), local.shape)
    max_lon, max_lat, max_speed = clon[imax[1]], clat[imax[0]], cspd[imax]
    ax2.scatter([max_lon], [max_lat], marker="*", s=190, color="gold",
                edgecolor="#6b5200", linewidth=1.0, zorder=15)
    ax2.annotate(
        f"Local current max ({max_speed:.2f} m s$^{{-1}}$)",
        xy=(max_lon, max_lat),
        xytext=(max_lon + 0.55, max_lat - 0.55),
        fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#8b6f00",
                  lw=0.7, alpha=0.95),
        arrowprops=dict(arrowstyle="-|>", color="#6b5200", lw=1.0),
        zorder=16,
    )
    ax2.set_xlim(WINDOW[:2])
    ax2.set_ylim(WINDOW[2:])
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.text(0.0, 1.035,
             "(b) Ocean current field (HYCOM, exampled by 30 m)",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=17, fontweight="bold")
    ax2.grid(color="0.85", alpha=0.25, lw=0.5)
    add_horizontal_colorbar(fig, cax2, im2,
                            "30 m current speed (m s$^{-1}$)")

    out_png = HERE / "figures" / "fig2_surface_subsurface_observation.png"
    out_pdf = HERE / "figures" / "fig2_surface_subsurface_observation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
