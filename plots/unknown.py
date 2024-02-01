from datetime import datetime
from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.interpolate import griddata


def plot_unknown(
    fn_time,
    time: datetime,
    # Coordenadas da área da figura [W, E, S, N]
    region: Tuple[float, float, float, float],
    # Longitude do nc de superficie
    lon: list,
    # Latitude do nc de superficie
    lat: list,
    # Coordenada de longitude do ponto de vazamento
    center_lon:float ,
    # Coordenada de latitude do ponto de vazamanto
    center_lat:float ,
    # Valores de espessura de óleo em superfície
    thick,
    # Arquivo nc de vento
    wind_file,
    perbio,
    persub,
    pere,
    perm,
    persed,
    perstr,
    lonrm,
    latrm,
    navg,
    levels,
    fig_dates_BRT,
    current_vectors_resolution: int,
    colors: list,
    output_file: str,
):
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_extent(region, crs=ccrs.PlateCarree())

    # ! WHAT???
    av_color = np.array([255 / 256, 199 / 256, 127 / 256, 1])
    av_color = np.array([av_color])

    av_color.repeat(1, 1).repeat(256, 0)
    ax1.pcolor(
        lon,
        lat,
        np.nanmax(thick[:fn_time, :], axis=0) / np.nanmax(thick[:fn_time, :], axis=0),
        color="grey",
        zorder=3,
    )
    ax1.plot(
        center_lon,
        center_lat,
        marker="s",
        markersize=5,
        color="brown",
        label="Oil on coast",
    )
    ax1.plot(
        center_lon,
        center_lat,
        marker="s",
        markersize=5,
        color="grey",
        label="Impacted area",
    )

    lonm2, latm2 = np.meshgrid(navg.lon.values, navg.lat.values)
    uo = navg.u_eastward.values
    vo = navg.v_northward.values
    ax1.coastlines(linewidth=0.8, color="black", zorder=4)
    ax1.add_feature(cfeature.LAND, zorder=3)

    uo = griddata(
        (lonm2.ravel(), latm2.ravel()), uo[0, :].ravel(), (lonrm.ravel(), latrm.ravel())
    ).reshape(lonrm.shape)
    vo = griddata(
        (lonm2.ravel(), latm2.ravel()), vo[0, :].ravel(), (lonrm.ravel(), latrm.ravel())
    ).reshape(lonrm.shape)
    aa = ax1.quiver(
        lonrm[0:-1:current_vectors_resolution, 0:-1:current_vectors_resolution],
        latrm[0:-1:current_vectors_resolution, 0:-1:current_vectors_resolution],
        uo[0:-1:current_vectors_resolution, 0:-1:current_vectors_resolution],
        vo[0:-1:current_vectors_resolution, 0:-1:current_vectors_resolution],
        width=0.0020,
        headwidth=6,
        linewidths=2,
        headlength=3,
        headaxislength=3,
        scale=8,
        color="k",
        alpha=0.5,
        zorder=1,
    )
    ax1.quiverkey(aa, 1.15, 0.68, 0.5, "Currents (0.5 m/s)", coordinates="axes")
    dx = geodesic((-20, -38.5), (-20, -38.0)).km
    scalebar = ScaleBar(dx, "km", length_fraction=0.25)
    ax1.add_artist(scalebar)

    plt.title(
        "Contínuo  " + fig_dates_BRT[fn_time].strftime("%Y/%m/%d - %H:%M"), size=15
    )

    legend1 = ax1.legend(loc=3, fontsize="medium")
    plt.gca().add_artist(legend1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = [
        "{} - {}".format(levels[i], levels[i + 1]) for i in range(len(levels) - 1)
    ]
    labels[-1] = "> {}".format(levels[-2])
    left, bottom, width, height = [0.72, 0.33, 0.09, 0.29]
    axins8 = fig.add_axes([left, bottom, width, height])
    axins8.legend(handles, labels, title="Thickness (\u03bc m)")
    axins8.set_xticks([])
    axins8.set_yticks([])
    axins8.axis("off")
    u_wind = wind_file.Uwind.sel(
        lon=center_lon,
        lat=center_lat,
        time=time,
        method="nearest",
    ).values
    v_wind = wind_file.Vwind.sel(
        lon=center_lon,
        lat=center_lat,
        time=time,
        method="nearest",
    ).values
    wmag = np.sqrt((u_wind) ** 2 + (v_wind) ** 2)
    left, bottom, width, height = [0.72, 0.73, 0.15, 0.15]
    axins = fig.add_axes([left, bottom, width, height])
    aa2 = axins.quiver(0.5, 0.5, u_wind, v_wind, headwidth=10, scale=35)
    axins.quiverkey(
        aa2,
        0.5,
        0.7,
        0,
        "Wind - " + str(np.round(wmag, 2)) + " m/s",
        coordinates="axes",
    )
    axins.set_xticks([])
    axins.set_yticks([])
    gl = ax1.gridlines(draw_labels=True, alpha=0.8, color="0.4", linestyle="dotted")
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}
    left, bottom, width, height = [0.72, 0.13, 0.15, 0.15]
    axins2 = fig.add_axes([left, bottom, width, height])
    process = ("Decayed", "Evaporated", "Surface", "Submerged", "Sedimented", "Coast")
    y_pos = np.arange(len(process))
    values = [perbio, pere, perm, persub, persed, perstr]
    axins2.barh(y_pos, values, align="center")
    axins2.grid(linestyle="-")
    axins2.set_yticks(y_pos)
    axins2.set_yticklabels(process)
    axins2.invert_yaxis()  # labels read top-to-bottom
    axins2.set_xlabel("%")
    axins2.yaxis.set_label_position("right")
    axins2.yaxis.tick_right()
    axins2.set_title("Mass Balance")
    axins2.set_xlim([0, 100])

    plt.savefig(output_file, dpi=200, transparent=False, bbox_inches="tight")
    plt.close()
