import os
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader
from matplotlib import dates
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset

import plots

matplotlib.use("TKAgg")
plt.style.use("default")

# =========================================== Sessao do Usuario ==========================================================================================================
USER_PROFILE_PATH = os.environ["USERPROFILE"]
BASE_PATH = r'C:\Users\nandara.bortoli\Downloads\ACU_VAST'

## Arquivos de saida do OSCAR (estes arquivos estao localizados na pasta Modelout do projeto)
# --------------------------------------------------------------------------------------------
# Caminho do arquivo ".nc" de superfície
surf_nc = os.path.join(BASE_PATH,"Modelout","ACU_VAST_periodo1_6_surface.nc")

# Caminho do arquivo ".nc" de toque de costa
shore_nc =  os.path.join(BASE_PATH,"Modelout","ACU_VAST_periodo1_6_shoreline.nc")

# Caminho do arquivo ".dto"
csv_file =os.path.join(BASE_PATH,"Modelout","ACU_VAST_periodo1_6.dto")
# --------------------------------------------------------------------------------------------
# Caminho para salvar os arquivos de saída
output_dir = os.path.join(BASE_PATH,'output')

os.makedirs(output_dir, exist_ok=True)

# Caminho do shape de linha de costa
shapefile = os.path.join(r"C:\Users\nandara.bortoli\OceanPact Serviços Marítimos S.A\ModOleo e qualidade de água - Documentos\General\Data\coastal_shp","BR_MAREM_OP_pol_v2.shp")

# Caminho do arquivo de corrente utilizado como input do OSCAR
cur_nc = os.path.join(BASE_PATH,"Currents","teste_new_cur_30112023.nc")

# Caminho do arquivo de vento utilizado como input do OSCAR
windnc = os.path.join(BASE_PATH,"Winds","oscar_wind_20231130.nc")

# Coordenada de longitude do ponto de vazamento
centerlon = -40.981000

# Coordenada de latitude do ponto de vazamanto
centerlat = -21.800833

# Distancias extremas do mapa considerando a lat e lon do ponto de vazamento (zoom geral do mapa)
dist = 2

# Resolução vetores corrente
sp = 70

# Tempos de saída para cada figura da simulação do vazamento
time_out = [0,1,2,3,4,5,6,7,8,9,10,11,12]

# Coordenadas da área da figura [W, E, S, N]
region = (-41.11, -40.75, -21.99, -21.68)
# ===========================================================================================================================================================

# Ler arquivo dto
csv = pd.read_csv(csv_file)

# Ler arquivo nc de superficie
surf_file = xr.open_dataset(surf_nc)

# Extrair coordenadas de latitude e longitude do nc de superficie
lon = surf_file.longitude.values
lat = surf_file.latitude.values

# Ler arquivo nc de linha de costa
shore_file = xr.open_dataset(shore_nc)

# Ler arquivo de corrente
curfile = xr.open_mfdataset(cur_nc)

# Ler arquivo nc de vento
windf = xr.open_mfdataset(windnc)

# Criar grade com latitudes e longitudes
lon, lat = np.meshgrid(lon, lat)

# Valores de espessura de óleo em superfície
thick = surf_file.surface_oil_thickness.values

# Valores de espessura de óleo na linha de costa
thicks = shore_file.oil_mass_per_unit_area_on_shoreline.values

# Copiar variavel de espessura de oleo em superficie e trocar os valores de nan por zero
thickco = thick.copy()
thickco[np.isnan(thickco)] = 0

# Criar e salvar tabelas csv e shapefile para cada tempo de saida selecionado com lons, lats e respectivas espessuras de óleo para cada ponto da grade
for iuy in time_out:
    data = {"lon": lon.ravel(), "lat": lat.ravel(), "thick": thick[iuy, :].ravel()}
    df = pd.DataFrame(data)
    df.to_csv(
        output_dir + "\\" + "thick_" + str(iuy + 1) + ".csv",
        index=False,
        sep=";",
        float_format="%.5f",
    )
    df.fillna(0, inplace=True)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"], df["thick"])
    )
    gdf.to_file(
        output_dir + "\\" + "output" + str(iuy + 1) + ".shp",
        driver="ESRI Shapefile",
        crs="EPSG:4326",
    )


shp = list(shapereader.Reader(shapefile).geometries())

latrm = np.arange(centerlat - dist, centerlat + dist, np.diff(lat[:, 0])[0])
lonrm = np.arange(centerlon - dist, centerlon + dist, np.diff(lat[:, 0])[0])
lonrm, latrm = np.meshgrid(lonrm, latrm)

begindate = Dataset(surf_nc)["time"].units[14:]
begindate = dates.datestr2num(begindate)
time = Dataset(surf_nc)["time"][:] / (24 * 60 * 60)
figdates = dates.num2date(begindate + time)

figdates_BRT = np.array(figdates) - timedelta(hours=3)

time = time * 24

levels = np.array([0.1, 0.6, 1, 10, 50, 200, 100000])

cmap = plt.cm.get_cmap("Paired")

colors = list(map(cmap, range(len(levels[0:-1]))))

cmap = plt.get_cmap("Paired")

norm = BoundaryNorm(levels[0:-1], ncolors=len(levels[0:-1]), clip=True)


black = False  # slick black
iuo = 0
jkt = 0
for tim in time_out:
    ####### DEFINITIONS ########
    year = figdates[tim].year
    month = figdates[tim].month
    day = figdates[tim].day
    hour = figdates[tim].hour
    minute = figdates[tim].minute
    second = figdates[tim].second
    time = datetime(year, month, day, hour, minute, second)
    navg = curfile.interp(TIME=time)
    perbio = csv.loc[tim + 1, " biodegraded(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100
    persub = csv.loc[tim + 1, " submerged(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100
    pere = csv.loc[tim + 1, " atmosphere(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100
    perm = csv.loc[tim + 1, " surface(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100
    persed = csv.loc[tim + 1, " sediment(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100
    perstr = csv.loc[tim + 1, " stranded(mt)"] / np.sum(csv.iloc[tim + 1, 1:9]) * 100

    ###########################################################################################
    plots.plot_unknown(
        fn_time=tim,
        time=time,
        # Coordenadas da área da figura [W, E, S, N]
        region=region,
        # Longitude do nc de superficie
        lon=lon,
        # Latitude do nc de superficie
        lat=lat,
        # Coordenada de longitude do ponto de vazamento
        center_lon=centerlon,
        # Coordenada de latitude do ponto de vazamanto
        center_lat=centerlat,
        # Valores de espessura de óleo em superfície
        thick=thick,
        # Arquivo nc de vento
        wind_file=windf,
        perbio=perbio,
        persub=persub,
        pere=pere,
        perm=perm,
        persed=persed,
        perstr=perstr,
        lonrm=lonrm,
        latrm=latrm,
        navg=navg,
        levels=levels,
        fig_dates_BRT=figdates_BRT,
        current_vectors_resolution=sp,
        colors=colors,
        output_file=os.path.join(
             output_dir, f"saida_nocleacc{time.strftime('%Y-%m-%d-%H-%M')}.png"
        ),
    )


viscmin = csv[" viscmin(mPa*s)"]
viscmean = csv[" viscmean(mPa*s)"]
viscmax = csv[" viscmax(mPa*s)"]
time = csv["time(days)"]


plots.plot_viscosity(
    time=time,
    visc_min=viscmin,
    visc_max=viscmax,
    visc_mean=viscmean,
    output_file=os.path.join(output_dir, "viscosity_prev.png"),
)


number = csv.shape[0]

time = csv["time(days)"]

perct = np.zeros(number)
perbt = np.zeros(number)
peret = np.zeros(number)
permt = np.zeros(number)
persubt = np.zeros(number)
percsed = np.zeros(number)

for tim in range(number):
    massb = csv.loc[tim, " biodegraded(mt)"]
    masse = csv.loc[tim, " atmosphere(mt)"]
    masss = csv.loc[tim, " surface(mt)"]
    massub = csv.loc[tim, " submerged(mt)"]
    masc = csv.loc[tim, " stranded(mt)"]
    massed = csv.loc[tim, " sediment(mt)"]
    masstot = csv.iloc[tim, 1:9].sum()
    perbt[tim] = (massb / masstot) * 100
    peret[tim] = (masse / masstot) * 100
    permt[tim] = (masss / masstot) * 100
    persubt[tim] = (massub / masstot) * 100
    perct[tim] = (masc / masstot) * 100
    percsed[tim] = (massed / masstot) * 100


limn = 70
limn = csv.shape[0]
timem = time
fig, axs = plt.subplots(1, figsize=(8, 5))
axs.plot(timem[0:limn], perbt[0:limn], linewidth=2, label="Decay")
axs.plot(timem[0:limn], peret[0:limn], linewidth=2, label="Evaporation")
axs.plot(timem[0:limn], permt[0:limn], linewidth=2, label="Surface")
axs.plot(timem[0:limn], persubt[0:limn], linewidth=2, label="Dispersion")
axs.plot(timem[0:limn], perct[0:limn], linewidth=2, label="Coast")
axs.plot(timem[0:limn], percsed[0:limn], linewidth=2, label="Sediment")
legend = axs.legend(bbox_to_anchor=(0.9, -0.12), loc=1, fontsize="small", ncol=5)
legend.get_frame().set_facecolor("grey")
axs.tick_params(labelsize=8)
fig.text(
    0.06,
    0.5,
    "Mass balance (%)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=11,
)
axs.set_xlabel("Time in days", fontsize=10)
axs.set_ylim([0, 100])
plt.savefig(
    output_dir + "\\" + "mass_balance_prev.png",
    dpi=200,
    transparent=False,
    bbox_inches="tight",
)
plt.close()


# tabela
rang = 2  # diminuir numero de linhas na tabela

idex = np.arange(0, surf_file.surface_oil_thickness.shape[0], rang)
if idex[-1] < surf_file.surface_oil_thickness.shape[0] - 1:
    idex[-1] = surf_file.surface_oil_thickness.shape[0] - 1

idex = np.array([5, 11, 23, 35, 59, 71])

thick = surf_file.surface_oil_thickness.isel(time=idex).values

R = 6373.0

thickmasked = np.ma.masked_invalid(thick)

datest = []
for i in range(len(figdates_BRT)):
    datest.append(figdates_BRT[i].strftime("%m/%d/-%H"))

datest = np.array(datest)
datest[idex]

table = csv.loc[
    :,
    [
        "time(days)",
        " surface(mt)",
        " biodegraded(mt)",
        " atmosphere(mt)",
        " submerged(mt)",
        " stranded(mt)",
        " outside(mt)",
        " sediment(mt)",
    ],
]
table = table.iloc[idex + 1]
table.insert(1, "Mass center lon", np.zeros(len(table)))
table.insert(1, "Mass center lat", np.zeros(len(table)))
table.insert(1, "Date(M/D-H)", datest[idex])
table.insert(3, "Distance to coast (Km)", np.zeros(len(table)))
table.insert(4, "Max conc. (ppm)", np.zeros(len(table)))
table.insert(
    table.shape[1], "Oil on surface(m3))", csv.iloc[idex][" vol th. oil(m3)"].values
)
table.loc[1, "Mass center lon"] = np.round(centerlon, 2)
table.loc[1, "Mass center lat"] = np.round(centerlat, 2)
table.loc[1, "Mass center lon"] = np.nan
table.loc[1, "Mass center lat"] = np.nan
loncoast = -39.8659
latcoast = -19.7119

lat1 = radians(latcoast)
lon1 = radians(loncoast)
lat2 = radians(centerlat)
lon2 = radians(centerlon)
dlon = lon2 - lon1
dlat = lat2 - lat1
a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
c = 2 * atan2(sqrt(a), sqrt(1 - a))
distance = R * c  # km
table.loc[1, "Distance to coast (Km)"] = np.round(distance, 2)

lat1 = radians(latcoast)
lon1 = radians(loncoast)
table.reset_index(inplace=True)


for i in range(0, len(table) - 1):
    table.loc[
        i,
        [
            " surface(mt)",
            " biodegraded(mt)",
            " atmosphere(mt)",
            " submerged(mt)",
            " stranded(mt)",
            " outside(mt)",
            " sediment(mt)",
        ],
    ] = (
        table.loc[
            i,
            [
                " surface(mt)",
                " biodegraded(mt)",
                " atmosphere(mt)",
                " submerged(mt)",
                " stranded(mt)",
                " outside(mt)",
                " sediment(mt)",
            ],
        ]
        / csv.iloc[idex[i] + 1, 1:9].sum()
        * 100
    )
    try:
        lni, lti = np.where(thickmasked[i] == thickmasked[i].max())
        table.loc[i, "Mass center lon"] = np.round(lon[lni, lti], 2)
        table.loc[i, "Mass center lat"] = np.round(lat[lni, lti], 2)
        table.loc[i, "Max thick. (micro.)"] = thickmasked[i].max()
        lat2 = radians(table.loc[i, "Mass center lat"])
        lon2 = radians(table.loc[i, "Mass center lon"])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c  # km
        table.loc[i, "Distance to coast (Km)"] = np.round(distance, 2)
    except:
        table.loc[i, "Distance to coast (Km)"] = np.nan
        table.loc[i, "Mass center lon"] = np.nan
        table.loc[i, "Mass center lat"] = np.nan
        table.loc[i, "Max thick. (micro.)"] = np.nan


table["time(days)"] = table["time(days)"] * 24
table.rename(
    columns={
        " biodegraded(mt)": "Decay (%)",
        " atmosphere(mt)": "Oil evaporated (%)",
        " submerged(mt)": "Oil in column(%)",
    },
    inplace=True,
)
table.rename(
    columns={
        " stranded(mt)": "Oil on coast (%)",
        " surface(mt)": "Oil on surface (%)",
        "time(days)": "time(h)",
    },
    inplace=True,
)
table.rename(
    columns={" outside(mt)": "Dispersed (%)", " sediment(mt)": "Sediment (%)"},
    inplace=True,
)
table["time(h)"] = np.round(table["time(h)"], 2)
table.round(2)

table.round(2).to_excel(output_dir + "output5.xlsx", index=False, sheet_name="first")
