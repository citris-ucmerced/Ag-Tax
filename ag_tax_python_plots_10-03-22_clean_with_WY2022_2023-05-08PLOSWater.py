import os
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ranksums
from scipy.stats import linregress

matplotlib.rcParams['contour.linewidth'] = 0.4
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.linewidth'] = 0.4 

# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

data_folder = "data_update_2023"

pr_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_pr_wy_by_DAU.csv"), delimiter=",")  # mm
etc_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_etc_wy_by_DAU.csv"), delimiter=",")  # mm
tmean_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_tmean_wy_by_DAU.csv"), delimiter=",")  # mean air temperature in Celsius
rmin_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_rmin_wy_by_DAU.csv"), delimiter=",")  # mean minimum relative humidity in %
rmax_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_rmax_wy_by_DAU.csv"), delimiter=",")  # mean maximum relative humidity in %
vpd_wy = np.loadtxt(os.path.join(data_folder, "CVavgYly_vpd_wy_by_DAU.csv"), delimiter=",")  # mean vapor pressure deficit in kPa

# = pd.DataFrame({'pr':[156],'etc':[1134]}) # eto is the average of the 
#etc_wy_with_2022 = etc_wy
#pr_wy_with_2022 = pr_wy

DEBUG = False   # adds extra bits to plotting to make sure that regression values are correct
START_WATER_YEAR = 1980
END_WATER_YEAR = 2023
units = 'SI'

MIN_BASELINE_YEAR = 1980
MAX_BASELINE_YEAR = 2011
MIN_RECENT_YEAR = 2012
MAX_RECENT_YEAR = 2023

# seaborn.regplot params. Moved up here to where other constants are while we fiddle
CI = 95
ROBUST = False
BOOTSTRAP_SAMPLES = 1000

def year_index(year, start_year=START_WATER_YEAR, inclusive=False):
    """
        Doesn't really need to be a function, but will make some code below readable
    :param year: The year to get the index value for
    :param start_year: The baseline year in the data that is the 0 index
    :param inclusive:  Whether to add 1 to the index to capture the end year for array slicing
    :return: index value in a sequential list/np array of annual data
    """

    index = int(year) - int(start_year)
    if inclusive:
        index += 1

    return index


lower_index = 0
upper_index = year_index(END_WATER_YEAR, inclusive=True)

print(f"Debug: Lower Index: {lower_index}, Upper Index: {upper_index}. Index range: {upper_index - lower_index}")

def data_year_slice(data, year, as_series=True):
    """
        Pulls out a slice of data from a numpy array corresponding to a single year. This could all be handled more
        efficiently, but working with the code as-written, this way improves it a bunch already.
    :param data: a 1-dimensional numpy array with annual values aligning to the time range of the rest of the code.
    :param year: the year to pull out - it will calculate the expected index of that year based on START_WATER_YEAR
    :param as_series: whether to coerce it to a pandas Series - by default it does because that's how it's used in other parts of the code.
    :return:
    """
    index = year_index(year)
    slice = data[index:index+1]
    if as_series:
        return pd.Series(slice)
    else:
        return slice


min_baseline_year_index = year_index(MIN_BASELINE_YEAR)
max_baseline_year_index = year_index(MAX_BASELINE_YEAR, inclusive=True)
min_recent_year_index = year_index(MIN_RECENT_YEAR)
max_recent_year_index = year_index(MAX_RECENT_YEAR, inclusive=True)
print(f"length of baseline years = {max_baseline_year_index - min_baseline_year_index}")
print(f"length of recent years = {max_recent_year_index - min_recent_year_index}")

if units == 'SI':
    etc_wy = etc_wy
    pr_wy = pr_wy
    tmean_wy = tmean_wy - 273.15  # convert from Kelvin to Celsius
elif units == 'english':
    etc_wy = etc_wy / 25.4
    pr_wy = pr_wy / 25.4
    tmean_wy = (tmean_wy - 273.15) * (9 / 5) + 32  # convert from Kelvin to Fahrenheit


etc_wy = etc_wy[lower_index:upper_index+1]
pr_wy = pr_wy[lower_index:upper_index+1]
tmean_wy = tmean_wy[lower_index:upper_index+1]
vpd_wy = vpd_wy[lower_index:upper_index+1]
rmin_wy = rmin_wy[lower_index:upper_index+1]
rmax_wy = rmax_wy[lower_index:upper_index+1]

# make an array of water years
wy = np.arange(START_WATER_YEAR, END_WATER_YEAR+1)  # water years

# definition to label points on the scatterplot 
def label_point(x, y, val, ax, xoffset, yoffset):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    b = pd.concat({'x': x+xoffset, 'y': y+yoffset, 'val': val}, axis=1)
    for i, point in a.iterrows():
        t = ax.text(point['x']+xoffset, point['y']+yoffset, str(round(point.val)))    
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='round', pad=0))
    c = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    c = pd.concat([c, b], axis=0, ignore_index = True)
    c.plot(x="x", y="y", kind="line", ax=ax, color='black', alpha=1, linestyle = 'dashed', zorder=1, label='_nolegend_',linewidth=0.05)

def label_point_english(x, y, val, ax, xoffset, yoffset):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    b = pd.concat({'x': x+xoffset, 'y': y+yoffset, 'val': val}, axis=1)
    for i, point in a.iterrows():
        t = ax.text(point['x']+xoffset, point['y']+yoffset, str(round(point.val)))    
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    c = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    c = pd.concat([c, b], axis=0, ignore_index = True)
    c.plot(x="x", y="y", kind="line", ax=ax, color='k', alpha=0.4, linestyle = 'dashed', zorder=1, label='_nolegend_')

def label_point_contour(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        t = ax.text(point['x'], point['y'], str(round(point.val))+'%')  
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', boxstyle='round', pad=0.1))

def label_point_no_leaders(x, y, val, ax, xoffset, yoffset):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        t = ax.text(point['x']+xoffset, point['y']+yoffset, str(round(point.val)))    
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

plt.rc('font', size=6)
plt.rc('axes', labelsize=6) 
plt.rc('xtick', labelsize=6) 
plt.rc('ytick', labelsize=6) 
plt.rc('legend', fontsize=5)
plt.rcParams ['font.family'] = 'Roboto'



# ETc versus Precipitation (mm)
fig1=plt.figure(figsize=(3.46,3.46))
fig1.subplots_adjust(top=0.95, left=0.18)
ax1 = fig1.add_subplot(111)
#p1 = sn.kdeplot(x=pr_wy[0:43], y=etc_wy[0:43], levels=[0.25,0.5,1], ax=ax1, shade_lowest=False, zorder=1, label='_nolegend_', color='grey', linewidth=0.8, kwargs={"linewidths":0.2})
p2 = sn.scatterplot(x=pr_wy[lower_index:upper_index], y=etc_wy[lower_index:upper_index], ax=ax1, marker="o", edgecolor='black', hue=tmean_wy[lower_index:upper_index], legend=False, label='_nolegend_', palette='coolwarm', s=15, facecolor='grey', alpha=1, zorder=10)
p7 = sn.regplot(x=pr_wy[min_baseline_year_index:max_baseline_year_index],
                y=etc_wy[min_baseline_year_index:max_baseline_year_index],
                ax=ax1,
                fit_reg=True,
                robust=ROBUST,
                n_boot=BOOTSTRAP_SAMPLES,
                ci=CI,
                color='b',
                marker='None')



results_baseline = linregress(pr_wy[min_baseline_year_index:max_baseline_year_index], etc_wy[min_baseline_year_index:max_baseline_year_index]) # slope, intercept, r, p, se
results_recent = linregress(pr_wy[min_recent_year_index:max_recent_year_index], etc_wy[min_recent_year_index:max_recent_year_index]) # slope, intercept, r, p, se
print("Baseline Results")
print(results_baseline)

print("Recent Years Results")
print(results_recent)

rsquared_baseline = results_baseline[2]*results_baseline[2]
print(f"Baseline R-Squared: {rsquared_baseline}")
rsquared_recent = results_recent[2]*results_recent[2]
print(f"Recent Years R-Squared: {rsquared_recent}")

if DEBUG:  # we don't always want this as it modifies figures, but we can use it for debug
    # we're going to manually plot the regression that we just ran to make sure if fits with seaborn's regression plot,
    # which we're leaving in place because it includes a confidence interval.
    x = [0, 600]  # set x values for lines

    # get the y values - at 0 it's just the intercept, at 600 it's y = mx + b or regression slope * 600 + intercept
    y_baseline = [results_baseline.intercept, results_baseline.slope * 600 + results_baseline.intercept]
    y_recent = [results_recent.intercept, results_recent.slope * 600 + results_recent.intercept]
    sn.lineplot(x=x, y=y_baseline)   # just add the lines now - don't make them pretty for debug.
    sn.lineplot(x=x, y=y_recent)

# Note that the regression plots below don't rely on the above regressions - seaborn runs its own, but the debug code above
# confirms identical outputs - seaborn doesn't really make a guarantee of that.

# Put a legend in by manually drawing lines up high
p27 = sn.lineplot(x=[800,801], y=[1000,1001], color='b', label=f'Regression {MIN_BASELINE_YEAR} - {MAX_BASELINE_YEAR}', lw=0.8)  # (y={results_baseline.slope:.2f}x+{results_baseline.intercept:4.0f})', lw=0.8)  # legend
p28 = sn.lineplot(x=[800,801], y=[1000,1001], color='r', label=f'Regression {MIN_RECENT_YEAR} - {MAX_RECENT_YEAR}', lw=0.8)  # (y={results_recent.slope:.2f}x+{results_recent.intercept:4.0f})', lw=0.8)  # legend

# add the recent years regression
p18 = sn.regplot(x=pr_wy[min_recent_year_index:max_recent_year_index],
                 y=etc_wy[min_recent_year_index:max_recent_year_index],
                 ax=ax1,
                 fit_reg=True,
                 robust=ROBUST,
                 n_boot=BOOTSTRAP_SAMPLES,
                 ci=CI,
                 color='r',
                 marker='None')
ax1.xaxis.set_tick_params(width=0.2)
ax1.yaxis.set_tick_params(width=0.2)

#for n in np.arange(0,len(p1.lines),1):
#    p1.lines[n]._linewidth = 0.8
for n in np.arange(0,len(p7.lines),1):
    p7.lines[n]._linewidth = 0.8
for n in np.arange(0,len(p18.lines),1):
    p18.lines[n]._linewidth = 0.8
for n in np.arange(0,len(p27.lines),1):
    p27.lines[n]._linewidth = 0.8
for n in np.arange(0,len(p28.lines),1):
    p28.lines[n]._linewidth = 0.8

for pos in ['top','right']:
    plt.gca().spines[pos].set_visible(False)

#sn.scatterplot(x=etc_wy_with_2022[41:42],y=pr_wy_with_2022[42:43], ax=ax1, legend=False, label='_nolegend_', s=90, facecolor='yellow', alpha=0.9, zorder=1)
#sn.scatterplot(x=forecast.pr, y=forecast.etc, ax=ax1, legend=False, label='_nolegend_', s=90, facecolor='green', alpha=0.9, zorder=1)
#t = np.linspace(0, 2*pi, 100) # ellipse
#plt.plot(forecast.pr[0] + 326.81*np.cos(t), forecast.etc[0] + 65.323*np.sin(t), axes=ax1) # ellipse
#plt.plot([pr_wy_with_2022[42:43],pr_wy_with_2022[42:43]],[1068.8,1199.5], 'g', axes=ax1, linewidth=1)
#plt.plot([pr_wy_with_2022[42:43]-5,pr_wy_with_2022[42:43]+5],[1068.8,1068.8], 'g', axes=ax1, linewidth=1)
#plt.plot([pr_wy_with_2022[42:43]-5,pr_wy_with_2022[42:43]+5],[1199.5,1199.5], 'g', axes=ax1, linewidth=1)
#plt.plot([pr_wy_with_2022[42:43],pr_wy_with_2022[42:43]+114],[forecast.etc,forecast.etc], 'g', axes=ax1, linewidth=1)
#plt.plot([pr_wy_with_2022[42:43]+114,pr_wy_with_2022[42:43]+114],[forecast.etc-3,forecast.etc+3], 'g', axes=ax1, linewidth=1)
#wy2022 = np.arange(1980, 2023) # water years
#label_point(pd.Series(pr_wy_with_2022[42:43]), forecast.etc, pd.Series(wy2022[42:43]), ax1, 28, 28) # 2021

#label_point_contour(pd.Series(140), pd.Series(1078), pd.Series(75), ax1) # contour 3

year_base_positions = {  # these are hardcoded based on Kelley's work - seems like they're tuned already and not an automatic lookup
    '2023': {'xoffset': 28, 'yoffset': 15},
    '2022': {'xoffset': -28, 'yoffset': 35},
    '2021': {'xoffset': -28, 'yoffset': 27},
    '2020': {'xoffset': 48, 'yoffset': 48},
    '2019': {'xoffset': 48, 'yoffset': 38},
    # '2018': {'xoffset': 18, 'yoffset': 50},
    '2017': {'xoffset': 90, 'yoffset': 8},
    '2016': {'xoffset': 28, 'yoffset': 28},
    '2015': {'xoffset': 98, 'yoffset': 48},
    '2014': {'xoffset': -60, 'yoffset': 35},
    '2013': {'xoffset': -94, 'yoffset': -70},
    # '2012': {'xoffset': 68, 'yoffset': 38},
    '2011': {'xoffset': -28, 'yoffset': -28},
    '1999': {'xoffset': -48, 'yoffset': -48},
    '1998': {'xoffset': -90, 'yoffset': -10},
    '1983': {'xoffset': 25, 'yoffset': 25},
}

if units == 'SI':
    for year in year_base_positions:
        label_point(data_year_slice(pr_wy, year), data_year_slice(etc_wy, year), data_year_slice(wy, year), ax1, **year_base_positions[year])
#    label_point_contour(pd.Series(226), pd.Series(970), pd.Series(25), ax1) # contour 1
#    label_point_contour(pd.Series(232), pd.Series(989), pd.Series(50), ax1) # contour 2
    #plt.xlim([40,600])
    plt.xlim([0,600])
    plt.ylim([890,1190])
    plt.xlabel('San Joaquin Valley Annual Total Precipitation (mm)')
    plt.ylabel('San Joaquin Valley Annual Total Crop Evapotranspiration (mm)')
    norm = plt.Normalize(tmean_wy.min(), tmean_wy.max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cb = ax1.figure.colorbar(sm, ticks=[16.3,16.6,16.9,17.2,17.5,17.8,18.1,18.4,18.7,18.9], orientation='vertical', pad=0.13, fraction=0.05)
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label('San Joaquin Valley Mean Annual Air Temperature (°C)')
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.2)
    fig1.savefig('__fig1_ETc_vs_Precip_mm.png', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_mm.svg', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_mm.eps', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_mm.pdf', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_mm.tiff', dpi=500)

elif units == 'english':

    # FIX - is there a way to iterate this?
    label_point_english(pd.Series(pr_wy[42:43]), pd.Series(etc_wy[42:43]), pd.Series(wy[42:43]), ax1, -28/25.4, 35/25.4) # 2022
    label_point_english(pd.Series(pr_wy[41:42]), pd.Series(etc_wy[41:42]), pd.Series(wy[41:42]), ax1, -28/25.4, 28/25.4) # 2021
    label_point_english(pd.Series(pr_wy[40:41]), pd.Series(etc_wy[40:41]), pd.Series(wy[40:41]), ax1, 38/25.4, 38/25.4) # 2020 
    label_point_english(pd.Series(pr_wy[39:40]), pd.Series(etc_wy[39:40]), pd.Series(wy[39:40]), ax1, 38/25.4, 38/25.4) # 2019
    label_point_english(pd.Series(pr_wy[38:39]), pd.Series(etc_wy[38:39]), pd.Series(wy[38:39]), ax1, 18/25.4, 36/25.4) # 2018
    label_point_english(pd.Series(pr_wy[37:38]), pd.Series(etc_wy[37:38]), pd.Series(wy[37:38]), ax1, 45/25.4, 38/25.4) # 2017
    label_point_english(pd.Series(pr_wy[36:37]), pd.Series(etc_wy[36:37]), pd.Series(wy[36:37]), ax1, 28/25.4, 28/25.4) # 2016
    label_point_english(pd.Series(pr_wy[35:36]), pd.Series(etc_wy[35:36]), pd.Series(wy[35:36]), ax1, -88/25.4, -58/25.4) # 2015
    label_point_english(pd.Series(pr_wy[34:35]), pd.Series(etc_wy[34:35]), pd.Series(wy[34:35]), ax1, -38/25.4, 38/25.4) # 2014
    label_point_english(pd.Series(pr_wy[33:34]), pd.Series(etc_wy[33:34]), pd.Series(wy[33:34]), ax1, -70/25.4, -50/25.4) # 2013
    #label_point_english(pd.Series(pr_wy[32:33]), pd.Series(etc_wy[32:33]), pd.Series(wy[32:33]), ax1, 38/25.4, 38/25.4) # 2012
    label_point_english(pd.Series(pr_wy[31:32]), pd.Series(etc_wy[31:32]), pd.Series(wy[31:32]), ax1, -18/25.4, -18/25.4) # 2011 
    label_point_english(pd.Series(pr_wy[19:20]), pd.Series(etc_wy[19:20]), pd.Series(wy[19:20]), ax1, -18/25.4, -18/25.4) # 1999
    label_point_english(pd.Series(pr_wy[18:19]), pd.Series(etc_wy[18:19]), pd.Series(wy[18:19]), ax1, 18/25.4, 18/25.4) # 1998 
    label_point_english(pd.Series(pr_wy[3:4]), pd.Series(etc_wy[3:4]), pd.Series(wy[3:4]), ax1, 18/25.4, 18/25.4) # 1983 
    label_point_contour(pd.Series(69/25.4), pd.Series(1095/25.4), pd.Series(25), ax1) # contour 1
    label_point_contour(pd.Series(95/25.4), pd.Series(1080/25.4), pd.Series(50), ax1) # contour 2
    plt.xlim([2,23])
    plt.xticks(np.arange(3, 24, step=3))
    plt.ylim([35,47])  #
    plt.xlabel('San Joaquin Valley Annual Total Precipitation (in)')
    plt.ylabel('San Joaquin Valley Annual Total Crop Evapotranspiration (in)')
    norm = plt.Normalize(tmean_wy.min(), tmean_wy.max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cb = ax1.figure.colorbar(sm, ticks=np.arange(tmean_wy.min(), tmean_wy.max(), step=0.4), orientation='vertical', pad=0.13)
    cb.set_label('San Joaquin Valley Mean Annual Air Temperature (°F)')
    fig1.savefig('__fig1_ETc_vs_Precip_in.png', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_in.svg', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_in.eps', dpi=500)
    fig1.savefig('__fig1_ETc_vs_Precip_in.tiff', dpi=500)

# ETc, climate variables
fig2, axes2 = plt.subplots(3, 1)
p50 = sn.lineplot(x=wy, y=etc_wy, color='g', markers=True, lw=0.8, ax=axes2[0]) 
p51 = sn.lineplot(x=wy, y=pr_wy, color='r', markers=True, lw=0.8, ax=axes2[0]) 
p52 = sn.lineplot(x=[2024,2026], y=[750,750], color='g', label='ETc', lw=0.8, ax=axes2[0]) # legend
p53 = sn.lineplot(x=[2024,2026], y=[750,750], color='r', label='Pr', lw=0.8, ax=axes2[0]) # legend
plt.xlim([1980,2022])  # FIXYEARS
plt.ylim([0,1200])  # FIXLIMITS
p50.set_xticks([1982,1986,1990,1994,
                1998,2002,2006,2010,
                2014,2018,2022])   # FIXYEARS
axes2[0].set_xlabel('Water-Year') 
axes2[0].set_ylabel('ETc and P (mm)')
p60 = sn.lineplot(x=wy, y=rmin_wy, color='m', markers=True, lw=0.8, ax=axes2[1]) 
p61 = sn.lineplot(x=wy, y=rmax_wy, color='b', markers=True, lw=0.8, ax=axes2[1], legend='auto') 
p60.set_xticks([1982,1986,1990,1994,
                1998,2002,2006,2010,
                2014,2018,2022])  # FIXYEARS
axes2[1].set_xlabel('Water-Year') 
axes2[1].set_ylabel('Relative Humidity (%)')
p70 = sn.lineplot(x=wy, y=tmean_wy, color='k', markers=True, lw=0.8, ax=axes2[2]) 
p70.set_xticks([1982,1986,1990,1994,
                1998,2002,2006,2010,
                2014,2018,2022])  # FIXYEARS
axes2[2].set_xlabel('Water-Year') 
axes2[2].set_ylabel('Mean Air Temp. (°F)')
fig2.savefig('__fig2_ETc_climate_timeseries.png', dpi=500)

# ETc, vpd
fig3=plt.figure(figsize=(3.46,3.46))
axes3 = fig3.add_subplot(111)
p80 = sn.scatterplot(x=vpd_wy, y=etc_wy, color='g', ax=axes3) 
axes3.set_ylabel('ETc (mm)')
axes3.set_xlabel('VPD (kPa)')
# linear regression for the above graph of vpd vs. etc
#result_lr = linregress(vpd_wy, etc_wy, alternative='two-sided')
    # result_lr: slope, intercept, rvalue, pvalue, stderr, intercept_stderr
#label_point_contour(pd.Series(69/25.4), pd.Series(1095/25.4), pd.Series(result_lr[]), axex3) 
#rsquared = result_lr[2]*result_lr[2]
fig3.savefig('__fig3_ETc_vs_vpd.png', dpi=500)
fig3.savefig('__fig3_ETc_vs_vpd.svg', dpi=500)
fig3.savefig('__fig3_ETc_vs_vpd.eps', dpi=500)


# # ETc, climate variables
# # Creating plot with dataset_1
# fig, ax1 = plt.subplots(3,1)
# color = 'tab:red'
# ax1.set_xlabel('X-axis')
# ax1.set_ylabel('Y1-axis', color = color)
# ax1.plot(wy, etc_wy, color = color)
# ax1.tick_params(axis ='y', labelcolor = color)
# # Adding Twin Axes to plot using dataset_2
# ax2 = ax1.twinx()
# color = 'tab:green'
# ax2.set_ylabel('Y2-axis', color = color)
# ax2.plot(wy, pr_wy, color = color)
# ax2.tick_params(axis ='y', labelcolor = color)
# # Adding title
# plt.title('Use different y-axes on the left and right of a Matplotlib plot', fontweight ="bold")
 
# # Show plot
# plt.show()

# Wilcoxon rank-sum test
result_etc = ranksums(etc_wy[0:31], etc_wy[32:43])  # FIXYEARS
result_pr = ranksums(pr_wy[0:31],pr_wy[32:43])  # FIXYEARS
result_vpd = ranksums(vpd_wy[0:31],vpd_wy[32:43])  # FIXYEARS




