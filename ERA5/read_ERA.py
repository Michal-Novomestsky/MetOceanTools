#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import netCDF4
import datetime
import numpy as np
from scipy.interpolate import RegularGridInterpolator

#metnc = netCDF4.Dataset('northRankinAMetRems-ws.2015.03.nc','r')
metnc = netCDF4.Dataset('with_dewpoint.nc','r')
time = np.array( metnc.variables['time'][:])        #Hours since 1900-01-01 00:00:00.0
lat = np.array(metnc.variables['latitude'][:])      #Latitude deg North
lon = np.array(metnc.variables['longitude'][:])     #Longitude deg East
u_10 = np.array(metnc.variables['u10'][:])          #10 metre U wind component (m/s)
v_10 = np.array(metnc.variables['v10'][:])          #10 metre V wind component (m/s)
two_m_temp = np.array(metnc.variables['t2m'][:])    #2 metre temperature (K)
two_m_dp = np.array(metnc.variables['d2m'][:])      #2 metre dewpoint (K)
mean_wave_dir = np.array(metnc.variables['mwd'][:]) #Mean wave direction in true deg (0deg North)
surface_temp = np.array(metnc.variables['sst'][:])  #Sea surface temperature (K)
surface_pres = np.array(metnc.variables['sp'][:])   #Surface pressure (Pa)
surface_solrad = np.array(metnc.variables['ssrd'][:]) #Surface solar radiation downwards (J/m^2)
surface_thermrad = np.array(metnc.variables['strd'][:]) #Surface thermal radiation downwards (J/m^2)
metnc.close()

latTrue = -19.5856
lonTrue = 116.1367

latIdx = 0
for i in range(len(lat)):
    if np.abs(lat[i] - latTrue) < np.abs(lat[latIdx] - latTrue):
        latIdx = i

lonIdx = 0
for i in range(len(lon)):
    if np.abs(lon[i] - lonTrue) < np.abs(lon[lonIdx] - lonTrue):
        lonIdx = i

timemet = []
u_10_met = []
v_10_met = []
two_m_temp_met = []
rh_met = []
spech_met = []
mean_wave_dir_met = []
surface_temp_met = []
surface_pres_met = []
surface_solrad_met = []
surface_thermrad_met = []

#rh coefficients
a = 17.62
b = 243.5

#spechum coefficients
Rdry = 287.0597
Rvap = 461.5250
a1 = 611.21
a3 = 17.502
a4 = 32.19
T0 = 273.15

print('Running through dates')
for i, hour in enumerate(time):
    start = datetime.datetime(year=1900, month=1, day=1, hour=0, minute=0, second=0)
    delta = datetime.timedelta(hours=int(hour+8)) #Moving ahead by 8hrs to go from GMT -> Perth time
    timemet.append(start + delta)

    #val[timeIdx][latIdx][lonIdx]: https://confluence.ecmwf.int/display/CKB/ERA5%3A+What+is+the+spatial+reference#ERA5:Whatisthespatialreference-Coordinatesystem
    #linearly interpolating to get point on 2D surface
    interp_u_10 = RegularGridInterpolator((lat, lon), u_10[i], bounds_error=True)
    u_10_met.append(interp_u_10((latTrue, lonTrue)))

    interp_v_10 = RegularGridInterpolator((lat, lon), v_10[i], bounds_error=True)
    v_10_met.append(interp_v_10((latTrue, lonTrue)))

    interp_two_m_temp = RegularGridInterpolator((lat, lon), two_m_temp[i], bounds_error=True)
    two_m_temp_met.append(interp_two_m_temp((latTrue, lonTrue)))

    interp_surf_pres = RegularGridInterpolator((lat, lon), surface_pres[i], bounds_error=True)
    surf_pres = interp_surf_pres((latTrue, lonTrue))
    surface_pres_met.append(surf_pres)

    #Calculating relative humidity (%): https://en.wikipedia.org/wiki/Dew_point
    interp_two_m_dp = RegularGridInterpolator((lat, lon), two_m_dp[i], bounds_error=True)
    T_dew = interp_two_m_dp((latTrue, lonTrue))
    ta = interp_two_m_temp((latTrue, lonTrue))
    T_dew -= 273.15 #Converting to degC for the formula
    ta -= 273.15
    exponent = (a*b*(T_dew - ta))/((b + ta)*(b + T_dew))
    rh_met.append(100*np.exp(exponent))

    #Calculating specific humidity (kg/kg)
    T_dew += 273.15
    E = a1*np.exp(a3*(T_dew-T0)/(T_dew-a4))
    spech_met.append((Rdry/Rvap)*E/(surf_pres-((1-Rdry/Rvap)*E)))

    interp_mean_wave_dir = RegularGridInterpolator((lat, lon), mean_wave_dir[i], bounds_error=True)
    mean_wave_dir_met.append(interp_mean_wave_dir((latTrue, lonTrue)))

    interp_surf_temp = RegularGridInterpolator((lat, lon), surface_temp[i], bounds_error=True)
    surface_temp_met.append(interp_surf_temp((latTrue, lonTrue)))

    interp_surf_solrad= RegularGridInterpolator((lat, lon), surface_solrad[i], bounds_error=True)
    surface_solrad_met.append(interp_surf_solrad((latTrue, lonTrue)))

    interp_surf_thermrad = RegularGridInterpolator((lat, lon), surface_thermrad[i], bounds_error=True)
    surface_thermrad_met.append(interp_surf_thermrad((latTrue, lonTrue)))

print('Saving')
np.savez('ERA5_2015', timemet=timemet, u_10=u_10_met, v_10=v_10_met, two_m_temp=two_m_temp_met, rh=rh_met, spechum=spech_met, mean_wave_dir=mean_wave_dir_met, surface_temp=surface_temp_met, surface_pres=surface_pres_met, surface_solrad=surface_solrad_met, surface_thermrad=surface_thermrad_met)