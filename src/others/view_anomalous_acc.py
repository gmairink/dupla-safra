#%% # LIBRARIES
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import math
import matplotlib.patches as patches
from matplotlib.patches import Patch

from matplotlib.colors import ListedColormap

path = "/home/gustavo/dupla_safra_view_web/data/processed/anomalous_accumulation"

baseval = 2.5
resolution = 0.5

#%% # Function panel configuration for first plot

def plot_background_plot_1(ax,nrows,ncols):

    #FUNCTION SETTINGS
    extent = [-67.25,-44.25,-22.75,-3.25]  #extent of projection [x0,x1,y0,y1] -> Brazil [-75,-32,-35,8]
    yticks = [-5, -10, -15, -20]  #tick marks drawn on y axis
    yticklabels = ['5°S', '10°S', '15°S', '20°S']  #ticks labels on y axis (same order as yticks)
    xticks = [-45, -50, -55, -60, -65]  #tick marks drawn on x axis
    xticklabels = ['45°W', '50°W', '55°W', '60°W', '65°W']  #ticks labels on x axis (same order as xticks)
    linewidth = 0.4

    #SETTINGS WITH VARIABLES OUTSIDE FUNCTION
    fontsize = plt.rcParams["font.size"]
    tick_fontsize = fontsize*0.8
    title_columns1 = panel_columns_title1
    title_columns2 = panel_columns_title2
    title_columns3 = panel_columns_title3
    title_columns4 = panel_columns_title4
    title_columns5 = panel_columns_title5
    title_columns6 = panel_columns_title6

    title_rows = panel_rows_title #tuple with titles

    #FUNCTION
    cont = 0
    for x in range(nrows):  #nrows of panel 
        for y in range(ncols):  #ncolumns of panel

            ax[x, y].set_extent(extent, ccrs.PlateCarree())
            ax[x, y].add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=linewidth)  # linha da costa
            ax[x, y].add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=linewidth)  # contorno dos países
            ax[x, y].add_feature(cfeature.STATES.with_scale('50m'), linewidth=linewidth)  # contorno dos estados do Brasil
            
            if y == 0:  #1st column
                ax[x,y].set_yticks(yticks)                   
                ax[x,y].set_yticklabels(yticklabels,fontsize=tick_fontsize)
                ax[x,y].set_ylabel(title_rows[x],fontsize=fontsize)
            else:  #other columns           
                ax[x,y].set_yticks(yticks)
                ax[x,y].set_yticklabels(yticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_ylabel('')
                
            if x == nrows-1:  #last row
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontsize=tick_fontsize)
                ax[x,y].set_xlabel('')
                ax[x,y].set_title(title_columns6[y], loc='center', fontsize=fontsize)
            elif x == 0:  #first row
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_xlabel('')
                ax[x,y].set_title(title_columns1[y], loc='center', fontsize=fontsize)
            else:    #other rows
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_xlabel('')
                
            if x == 1:     #second row
                ax[x,y].set_title(title_columns2[y], loc='center', fontsize=fontsize)
            elif x == 2:           #third row
                ax[x,y].set_title(title_columns3[y], loc='center', fontsize=fontsize)
            elif x == 3:          #fourth row
                ax[x,y].set_title(title_columns4[y], loc='center', fontsize=fontsize)
            elif x== 4:               #fifth row
                ax[x,y].set_title(title_columns5[y], loc='center', fontsize=fontsize)
    

            cont+=1
    return ax

#%% # Function panel configuration for second plot

def plot_background_plot_2(ax,nrows,ncols):

    #FUNCTION SETTINGS
    extent = [-67.25,-44.25,-22.75,-3.25]  #extent of projection [x0,x1,y0,y1] -> Brazil [-75,-32,-35,8]
    yticks = [-5, -10, -15, -20]  #tick marks drawn on y axis
    yticklabels = ['5°S', '10°S', '15°S', '20°S']  #ticks labels on y axis (same order as yticks)
    xticks = [-45, -50, -55, -60, -65]  #tick marks drawn on x axis
    xticklabels = ['45°W', '50°W', '55°W', '60°W', '65°W']  #ticks labels on x axis (same order as xticks)
    linewidth = 0.4

    #SETTINGS WITH VARIABLES OUTSIDE FUNCTION
    fontsize = plt.rcParams["font.size"]
    tick_fontsize = fontsize*0.8
    title_columns7 = panel_columns_title7
    title_columns8 = panel_columns_title8
    title_columns9 = panel_columns_title9
    title_columns10 = panel_columns_title10
    title_columns11 = panel_columns_title11
    # title_columns6 = panel_columns_title6

    title_rows = panel_rows_title #tuple with titles

    #FUNCTION
    cont = 0
    for x in range(nrows):  #nrows of panel 
        for y in range(ncols):  #ncolumns of panel

            ax[x, y].set_extent(extent, ccrs.PlateCarree())
            ax[x, y].add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=linewidth)  # linha da costa
            ax[x, y].add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=linewidth)  # contorno dos países
            ax[x, y].add_feature(cfeature.STATES.with_scale('50m'), linewidth=linewidth)  # contorno dos estados do Brasil
            
            if y == 0:  #1st column
                ax[x,y].set_yticks(yticks)                   
                ax[x,y].set_yticklabels(yticklabels,fontsize=tick_fontsize)
                ax[x,y].set_ylabel(title_rows[x],fontsize=fontsize)
            else:  #other columns           
                ax[x,y].set_yticks(yticks)
                ax[x,y].set_yticklabels(yticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_ylabel('')
                
            if x == nrows-1:  #last row
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontsize=tick_fontsize)
                ax[x,y].set_xlabel('')
                ax[x,y].set_title(title_columns11[y], loc='center', fontsize=fontsize)
            elif x == 0:  #first row
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_xlabel('')
                ax[x,y].set_title(title_columns7[y], loc='center', fontsize=fontsize)
            else:    #other rows
                ax[x,y].set_xticks(xticks)                    
                ax[x,y].set_xticklabels(xticklabels,fontdict={'fontsize':math.exp(-100),'color':'white'})
                ax[x,y].set_xlabel('')
                
            if x == 1:     #second row
                ax[x,y].set_title(title_columns8[y], loc='center', fontsize=fontsize)
            elif x == 2:           #third row
                ax[x,y].set_title(title_columns9[y], loc='center', fontsize=fontsize)
            elif x== 3:               #fourth row
                ax[x,y].set_title(title_columns10[y], loc='center', fontsize=fontsize)
    

            cont+=1
    return ax

#%% # Panel configurations

# para o primeiro plot
panel_columns_title1 = ('1981', '1982', '1983', '1984')
panel_columns_title2 = ('1985', '1986', '1987', '1988')
panel_columns_title3 = ('1989', '1990', '1991', '1992')
panel_columns_title4 = ('1993', '1994', '1995', '1996')
panel_columns_title5 = ('1997', '1998', '1999', '2000')
panel_columns_title6 = ('2001', '2002', '2003', '2004')

# para o segundo plot
panel_columns_title7 = ('2005', '2006', '2007', '2008')
panel_columns_title8 = ('2009', '2010', '2011', '2012')
panel_columns_title9 = ('2013', '2014', '2015', '2016')
panel_columns_title10 = ('2017', '2018', '2019', '2020')
panel_columns_title11 = ('2021', '2022', '2023', 'mean 1981-2023')

panel_rows_title = ("", "", "", "", "", "")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
fontsize = plt.rcParams["font.size"]

#%% # manipulação para garantir que todos os times (anos) contarão com dados com pelo menos 1 integrante de cada uma das classes da legenda
# RTA: recurso técnico alternativo

ds = xr.open_dataset(path+f'/demise_anomalous_accumulation_1981-2025.nc')
ds_aux = ds

longitude_target = -4.75
latitude_values = {
    -3.25: 190,
    -3.75: 200,
    -4.25: 220,
    -4.75: 235,
    -5.25: 245,
    -5.75: 265,
    -6.25: 280,
    -6.75: 295,
    -7.25: 310,
    -7.75: 325,
    -8.25: 340,
    -8.75: 180,
    -9.25: 200,
    -9.75: 220,
    -10.25: 235,
    -10.75: 245,
    -11.25: 265,
    -11.75: 280,
    -12.25: 295,
    -12.75: 310,
    -13.25: 325,
    -13.75: 340,
    -14.25: 180,
    -14.75: 200,
    -15.25: 220,
    -15.75: 235,
    -16.25: 245,
    -16.75: 265,
    -17.25: 280,
    -17.75: 295,
    -18.25: 310,
    -18.75: 325,
    -19.25: 340,
    -19.75: 180,
    -20.25: 200,
    -20.75: 360,
    -21.25: 235,
    -21.75: 245,
    -22.25: 265,
    -22.75: 210
}

# Adicionar a longitude -49.25 ao dataset, preservando as dimensões originais
new_longitudes = np.append(ds_aux['longitude'].values, longitude_target)

# Reindexar o dataset para incluir a nova longitude
ds_aux = ds_aux.reindex(longitude=new_longitudes, method=None)

# Preencher a nova longitude com valores NaN (vazios) inicialmente
ds_aux['demise'] = ds_aux['demise'].fillna(np.nan)


for time_val in ds_aux['time'].values:
    for lat, add_value in latitude_values.items():

        ds_aux['demise'].loc[
            dict(
                time=time_val,
                latitude=ds_aux['latitude'].sel(
                    latitude=lat,
                    method='nearest'
                ).item(),
                longitude=longitude_target
            )
        ] = add_value
        
        
        
# for time_val in ds_aux['time'].values:
#     for lat, add_value in latitude_values.items():
#         # Adicionar o valor para a latitude e longitude correspondentes
#         ds_aux['demise'].loc[dict(time=time_val, latitude=lat, longitude=longitude_target)] = add_value
        
#%% # painel para plot 4 colunas por 6 linhas

nrows,ncols = 6,4
xsize,ysize = 17,22 #x,y figure sizes

# infos da figura
fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(xsize,ysize),sharey=True,sharex=True
                            ,constrained_layout=True,subplot_kw={'projection':ccrs.PlateCarree()})

# cont para selecionar o ano corrente no loop
cont = 0

# array com ao anos para o plot
years = np.arange(1981,2005,1)

default_colors = ['#CC4C39', '#FD8C50', '#FBCF84', '#FFFFBD', '#8AC2E0', '#4769BA', '#31349C', '#8A2BE2', '#A8E6A2', '#4CAF50','#7F7F7F']

# loop para as linhas do sub-plot
for linha in range(nrows):
     # loop sob as colunas do sub-plot
    for coluna in range(ncols):  
        
        year = years[cont]
           
        # extrai os valores correspondentes ao ano corrente
        demise_plot = ds_aux['demise'].sel(time=year)
        demise = demise_plot.values
    

        # atribui os valores em classes, de acordo com o demise do pixel    
        february1 = (demise >= 215) & (demise < 230)
        february2 = (demise >= 230) & (demise < 243)
        march1 = (demise >= 243) & (demise < 258)
        march2 = (demise >= 258) & (demise < 274)
        april1 = (demise >= 274) & (demise < 289)
        april2 = (demise >= 289) & (demise < 304)
        may1 = (demise >= 304) & (demise < 319)
        may2 = (demise >= 319) & (demise < 335)
        june1 = (demise >= 335) & (demise < 350)
        june2 = (demise >= 350) & (demise <= 366)
        
        other = (demise <= 214)     # demise em dezembro 
        
    # Criar um array de cores baseado nos grupos
        color_data = np.zeros_like(demise, dtype=int)  
        color_data[february1] = 0 
        color_data[february2] = 1 
        color_data[march1] = 2 
        color_data[march2] = 3 
        color_data[april1] = 4  
        color_data[april2] = 5 
        color_data[may1] = 6
        color_data[may2] = 7 
        color_data[june1] = 8 
        color_data[june2] = 9 
        color_data[other] = 10 
                      
        current_cmap = ListedColormap(default_colors)
             
        # definir o subplot
        plot = axes[linha, coluna].pcolormesh(demise_plot.longitude, demise_plot.latitude, color_data, cmap=current_cmap, shading='nearest')
            
        cont += 1

# assetar fundo branco para a figura
fig.patch.set_facecolor('white')

# assetar legenda e cores para os boxes da legendas
legend_labels = [Patch(color=default_colors[i], label=f'February-1' if i == 0 else
                                      f'February-2' if i == 1 else
                                      f'March-1' if i == 2 else
                                      f'March-2' if i == 3 else
                                      f'April-1' if i == 4 else
                                      f'April-2' if i == 5 else
                                      f'May-1' if i == 6 else
                                      f'May-2' if i == 7 else
                                      f'June-1' if i == 8 else
                                      f'June-2' if i == 9 else
                                      'Other') for i in range(len(default_colors))]

fig.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.005),
           ncol=6,title=f'Rainy season demise (base value {baseval})', frameon=False,
           prop={'size': 12, 'family': 'serif'})

plot_background_plot_1(axes,nrows,ncols)

# fig.savefig(path+f'painel_onset_demise/demise_br_central_base_value_{baseval}_1981-2004.png', bbox_inches='tight',dpi=500)  # Ajuste o formato e os parâmetros conforme necessário

#%% # painel para plot 4 colunas por 5 linhas

nrows, ncols = 6, 4
xsize, ysize = 16, 22

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(xsize, ysize),
    sharey=True,
    sharex=True,
    constrained_layout=True,
    subplot_kw={'projection': ccrs.PlateCarree()}
)

cont = 0

years = np.arange(2005, 2026)

default_colors = [
    '#CC4C39',
    '#FD8C50',
    '#FBCF84',
    '#FFFFBD',
    '#8AC2E0',
    '#4769BA',
    '#31349C',
    '#8A2BE2',
    '#A8E6A2',
    '#4CAF50',
    '#7F7F7F'
]

for linha in range(nrows):

    for coluna in range(ncols):

        ax = axes[linha, coluna]

        if cont < len(years):

            year = years[cont]

            demise_plot = ds_aux['demise'].sel(time=year)

            demise = demise_plot.values

            title = str(year)

        else:

            demise_plot = ds['demise'].mean(dim='time')

            longitude_target = -43.75

            latitude_values = {
                -3.25: 190,
                -3.75: 200,
                -4.25: 220,
                -4.75: 235,
                -5.25: 245,
                -5.75: 265,
                -6.25: 280,
                -6.75: 295,
                -7.25: 310,
                -7.75: 325,
                -8.25: 340,
                -8.75: 180,
                -9.25: 200,
                -9.75: 220,
                -10.25: 235,
                -10.75: 245,
                -11.25: 265,
                -11.75: 280,
                -12.25: 295,
                -12.75: 310,
                -13.25: 325,
                -13.75: 340,
                -14.25: 180,
                -14.75: 200,
                -15.25: 220,
                -15.75: 235,
                -16.25: 245,
                -16.75: 265,
                -17.25: 280,
                -17.75: 295,
                -18.25: 310,
                -18.75: 325,
                -19.25: 340,
                -19.75: 180,
                -20.25: 200,
                -20.75: 220,
                -21.25: 235,
                -21.75: 245,
                -22.25: 265
            }

            new_longitudes = np.append(
                demise_plot['longitude'].values,
                longitude_target
            )

            demise_plot = demise_plot.reindex(
                longitude=new_longitudes,
                method=None
            )

            for lat, add_value in latitude_values.items():

                lat_real = demise_plot['latitude'].sel(
                    latitude=lat,
                    method='nearest'
                ).item()

                demise_plot.loc[
                    dict(
                        latitude=lat_real,
                        longitude=longitude_target
                    )
                ] = add_value

            demise = demise_plot.values

            title = 'mean 1981-2025'

        # =====================================================
        # CLASSIFICAÇÃO
        # =====================================================

        february1 = (demise >= 215) & (demise < 230)
        february2 = (demise >= 230) & (demise < 243)

        march1 = (demise >= 243) & (demise < 258)
        march2 = (demise >= 258) & (demise < 274)

        april1 = (demise >= 274) & (demise < 289)
        april2 = (demise >= 289) & (demise < 304)

        may1 = (demise >= 304) & (demise < 319)
        may2 = (demise >= 319) & (demise < 335)

        june1 = (demise >= 335) & (demise < 350)
        june2 = (demise >= 350) & (demise <= 366)

        other = (demise <= 214)

        color_data = np.zeros_like(
            demise,
            dtype=int
        )

        color_data[february1] = 0
        color_data[february2] = 1
        color_data[march1] = 2
        color_data[march2] = 3
        color_data[april1] = 4
        color_data[april2] = 5
        color_data[may1] = 6
        color_data[may2] = 7
        color_data[june1] = 8
        color_data[june2] = 9
        color_data[other] = 10

        current_cmap = ListedColormap(default_colors)

        # =====================================================
        # PLOT
        # =====================================================

        ax.pcolormesh(
            demise_plot.longitude,
            demise_plot.latitude,
            color_data,
            cmap=current_cmap,
            shading='nearest',
            transform=ccrs.PlateCarree()
        )

        ax.set_extent(
            [-67.25, -44.25, -22.75, -3.25],
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(
            cfeature.COASTLINE.with_scale('50m'),
            linewidth=0.4
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale('50m'),
            linewidth=0.4
        )

        ax.add_feature(
            cfeature.STATES.with_scale('50m'),
            linewidth=0.4
        )

        ax.set_title(
            title,
            fontsize=12
        )

        ax.set_xticks(
            [-45, -50, -55, -60, -65],
            crs=ccrs.PlateCarree()
        )

        ax.set_yticks(
            [-5, -10, -15, -20],
            crs=ccrs.PlateCarree()
        )

        ax.set_xticklabels(
            ['45°W', '50°W', '55°W', '60°W', '65°W'],
            fontsize=9
        )

        ax.set_yticklabels(
            ['5°S', '10°S', '15°S', '20°S'],
            fontsize=9
        )

        cont += 1


# =====================================================
# PAINEL VAZIO
# =====================================================

for i in range(cont, nrows * ncols):

    row = i // ncols
    col = i % ncols

    axes[row, col].set_visible(False)


# =====================================================
# FUNDO
# =====================================================

fig.patch.set_facecolor('white')


# =====================================================
# LEGENDA
# =====================================================

legend_labels = [
    Patch(color=default_colors[0], label='February-1'),
    Patch(color=default_colors[1], label='February-2'),
    Patch(color=default_colors[2], label='March-1'),
    Patch(color=default_colors[3], label='March-2'),
    Patch(color=default_colors[4], label='April-1'),
    Patch(color=default_colors[5], label='April-2'),
    Patch(color=default_colors[6], label='May-1'),
    Patch(color=default_colors[7], label='May-2'),
    Patch(color=default_colors[8], label='June-1'),
    Patch(color=default_colors[9], label='June-2'),
    Patch(color=default_colors[10], label='Other')
]

fig.legend(
    handles=legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.005),
    ncol=6,
    title='Rainy season demise',
    frameon=False,
    prop={'size': 12, 'family': 'serif'}
)

plt.show()

#%%

