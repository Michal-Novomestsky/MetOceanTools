import os
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import MeterologicalParameters.humidity as hum
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import sys
import argparse
import time

from scipy import integrate
from Modules.DataAnalyser import *
from COARE.COARE3_6.coare36vnWarm_et import coare36vnWarm_et as coare

# Defining constants
KELVIN_TO_CELSIUS = 273.15
ZU = 14.8 # Height of anemometer #1 (the higher one)
ZT = 28 # Approx. height of flare bridge AMSL
ZQ = 28 # Approx. height of flare bridge AMSL
LAT = -19.5856 # 19.5856S (Babanin et al.)
LON = 116.1367 # 116.1367E
SS = 35 # https://salinity.oceansciences.org/overview.htm
TIME_INTERVAL = 40

# Default parameters
LW_DN = 370
ZI = 600
RAINRATE = 0

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def analysis_loop(readDir: Path, eraDf: pd.DataFrame, remsDf: pd.DataFrame, supervised=True, cpuFraction=1, era_only=False, no_era=False) -> pd.DataFrame:
    """
    Steps through each data file located in readDir.
    Unsupervised enables multiprocessing using cpuFraction% of all available cores.
    """
    file_selector = lambda file: "NRAFBR" in file.stem
    files = [file for file in readDir.iterdir() if file_selector(file)]

    # TODO: Maybe make more efficient with smth like array. For now list is ok since we're passing in floats
    collector_time = []
    collector_tauApprox = []
    collector_tauCoare = []
    collector_HApprox = []
    collector_HCoare = []
    collector_Cd = []
    collector_U_10 = []
    collector_w_turb = []
    collector_u = []
    collector_v = []
    collector_w = []
    collector_t = []
    collector_rho = []
    collector_t1_fluct = []
    collector_t1_rng = []
    collector_t2_fluct = []
    collector_t2_rng = []

    # One-by-one
    if supervised:
        for file in files:
            output = _analysis_iteration(file, eraDf, remsDf, era_only, no_era)
            if output is not None:
                collector_tauApprox += output[0]
                collector_tauCoare += output[1]
                collector_Cd += output[2]
                collector_U_10 += output[3]
                collector_HApprox += output[4]
                collector_HCoare += output[5]
                collector_w_turb += output[6]
                collector_time += output[7]
                collector_u += output[8]
                collector_v += output[9]
                collector_w += output[10]
                collector_t += output[11]
                collector_rho += output[12]
                collector_t1_fluct += output[13]
                collector_t1_rng += output[14]
                collector_t2_fluct += output[15]
                collector_t2_rng += output[16]

    # Enabling multiprocessing
    else:
        if cpuFraction > 1 or cpuFraction <= 0:
            raise ValueError("cpuFraction must be between (0,1]")

        cpuCount = mp.cpu_count()
        coresToUse = int(np.ceil(cpuFraction*cpuCount))
        write_message(f"Using {100*cpuFraction}% of available cores -> {coresToUse}/{cpuCount}", filename='analysis_log.txt')

        # Creating a tuple of tuples of inputs to pass into each iteration
        remsArr = [remsDf]*len(files)
        eraArr = [eraDf]*len(files)
        eraOnlyArr = [era_only]*len(files)
        noEraArr = [no_era]*len(files)
        args = [*zip(files, eraArr, remsArr, eraOnlyArr, noEraArr)]

        with mp.Pool(coresToUse) as p:
            output = p.starmap(_analysis_iteration, iterable=args)
            for outputElem in output:
                if outputElem is not None:
                    collector_tauApprox += outputElem[0]
                    collector_tauCoare += outputElem[1]
                    collector_Cd += outputElem[2]
                    collector_U_10 += outputElem[3]
                    collector_HApprox += outputElem[4]
                    collector_HCoare += outputElem[5]
                    collector_w_turb += outputElem[6]
                    collector_time += outputElem[7]
                    collector_u += outputElem[8]
                    collector_v += outputElem[9]
                    collector_w += outputElem[10]
                    collector_t += outputElem[11]
                    collector_rho += outputElem[12]
                    collector_t1_fluct += outputElem[13]
                    collector_t1_rng += outputElem[14]
                    collector_t2_fluct += outputElem[15]
                    collector_t2_rng += outputElem[16]

    write_message("Analysis run done!", filename='analysis_log.txt')
    return pd.DataFrame({"time": collector_time, "tauApprox": collector_tauApprox, "tauCoare": collector_tauCoare,
                            "Cd": collector_Cd, "U_10": collector_U_10, "HApprox": collector_HApprox, "HCoare": collector_HCoare, 
                            "wTurb": collector_w_turb, "u": collector_u, "v": collector_v, "w": collector_w, "ta": collector_t,
                            "rho": collector_rho, "is_temp1_fluctuating": collector_t1_fluct, "is_temp1_range_large": collector_t1_rng, 
                            "is_temp2_fluctuating": collector_t2_fluct, "is_temp2_range_large": collector_t2_rng})

def _analysis_iteration(file: Path, eraDf: pd.DataFrame, remsDf: pd.DataFrame, era_only=False, no_era=False) -> None:
    """
    Internal function which runs an iteration of an analysis run. Iterated externally by analysis_loop.
    """
    fileName = file.stem
    date = fileName[7:15]
    day = date[0:2]
    month = date[2:4]
    hour = fileName[16:18]
    
    # Defining constants
    TS_DEPTH = remsDf.depth[0] # Note that depth has to be extracted before we select the corresponding day as sometimes REMS may not exist on that day

    # Getting the corresponding day in the REMS data
    remsDf = remsDf.loc[(remsDf.timemet.map(lambda x: x.day) == int(day)) & (remsDf.timemet.map(lambda x: x.hour) == int(hour)) & (remsDf.timemet.map(lambda x: x.month) == int(month))]
    remsDf = remsDf.reset_index()
    eraDf = eraDf.loc[(eraDf.timemet.map(lambda x: x.day) == int(day)) & (eraDf.timemet.map(lambda x: x.hour) == int(hour)) & (eraDf.timemet.map(lambda x: x.month) == int(month))]
    eraDf = eraDf.reset_index()

    # Getting TIME_INTERVAL minute long slices and using them to get turbulent avg data over that same time frame
    data = DataAnalyser(file)
    slices = get_time_slices(data.df, TIME_INTERVAL)

    # NOTE: FILL IN AS REQUIRED
    time_list = []
    tau_approx = []
    tau_coare = []
    H_approx = []
    H_coare = []
    C_d = []
    U_10_mag = [] # NOTE: "_mag" is to prevent it being const from all caps
    w_turb_list = []
    u_mean = []
    v_mean = []
    w_mean = []
    t_mean = []
    rho_mean = []
    is_temp1_fluctuating = []
    is_temp1_range_large = []
    is_temp2_fluctuating = []
    is_temp2_range_large = []

    w2 = "Anemometer #1 W Velocity (ms-1)"
    u2 = "Anemometer #1 U Velocity (ms-1)"
    v2 = "Anemometer #1 V Velocity (ms-1)"
    t2 = "Anemometer #1 Temperature (degC)"
    comp2 = "Compass #1 (deg)"
    
    if (era_only or len(remsDf) == 0) and not no_era:
        time = eraDf.timemet[0]
        era_and_rems = False
    elif len(remsDf) != 0:
        time = remsDf.timemet[0]
        era_and_rems = True
    # If there's no match with REMS and ERA5 isn't being used:
    elif no_era:
        write_message(f"No date matches between {fileName} and REMS. ERA5 turned off.", filename='analysis_log.txt')
        return None
    else:
        raise ValueError("None of the analyses cases were triggered")  

    for _, slice in enumerate(slices):
        # Using ERA5 data
        if not era_and_rems:
            dataSliceTemp = eraDf.loc[(time <= eraDf.timemet) & (eraDf.timemet <= time + datetime.timedelta(minutes=TIME_INTERVAL))]
            dataSliceTemp = dataSliceTemp.mean(numeric_only=True)
            if pd.notna(dataSliceTemp.loc['index']):
                dataSlice = dataSliceTemp # Guarding against ERA5's hour resolution from resulting in NaNs when incrementing up by less than 1hr at a time
        # Using REMS data
        else:
            dataSlice = remsDf.loc[(time <= remsDf.timemet) & (remsDf.timemet <= time + datetime.timedelta(minutes=TIME_INTERVAL))]
            dataSlice = dataSlice.mean(numeric_only=True)

        # TODO: Correcting for POSSIBLE error in anem temp (10degC hotter than REMS)
        #slice[t2] = slice[t2] - 5
        #slice[u2] = -slice[u2]
        #slice[v2] = -slice[v2]

        # Getting parameters
        jd = time - datetime.datetime(2015, 1, 1)
        jd = float(jd.days)
        tair = dataSlice.ta
        rh = dataSlice.rh
        p = dataSlice.press
        tsea = dataSlice.tsea
        sw_dn = dataSlice.solrad
        if not era_and_rems: lw_dn = dataSlice.thermrad # Only available with ERA5
        spechum = dataSlice.spech
        #e = hum.hum2ea_modified(p, spechum)
        rho = hum.rhov_modified(tair, p, sh=spechum)

        w2_turb = get_turbulent(slice[w2])
        T2_turb = get_turbulent(slice[t2])
        #T2_turb = T2_turb/(1 + 0.378*e/p)
        w_turb_list.append(np.mean(w2_turb))

        # Getting magnitude of turbulent horizontal velocity vector
        U2_turb = get_turbulent(np.sqrt(slice[u2]**2 + slice[v2]**2))

        # Getting current-corrected windspeed
        U2_mag = np.sqrt(slice[u2]**2 + slice[v2]**2)
        # Easterly -> +ive x axis, Northerly -> +ive y. Note that anem v+ is west so east is -v
        U2_vec = pd.DataFrame({'East': slice[v2], 'North': slice[u2]})
        U2_vec = U2_vec.mean() #Taking 10min avg

        # u_AirWat = u_Air - u_Wat
        U_vec = U2_vec
        #U_vec.East = U_vec.East - remsSlice.cur_e_comp # Seem to be negligible compared to wind speed
        #U_vec.North = U_vec.North - remsSlice.cur_n_comp
        u = np.sqrt(U_vec.North**2 + U_vec.East**2)

        u_star_2 = get_covariance(U2_turb, w2_turb)
        tau_approx.append(-rho*u_star_2)
        H_approx.append(rho*hum.cpd*get_covariance(w2_turb, T2_turb))

        #TODO: Assume U_10 ~= U_14.8 for now
        #C_d.append(np.mean(-U2_turb*w2_turb)/(np.mean(U2_mag)**2))
        C_d.append(u_star_2/(np.mean(U2_mag)**2))
        #C_d.append(-np.cov([U2_turb.mean(), w2_turb.mean()])/np.mean(U2_mag)
        U_10_mag.append(np.mean(U2_mag))
        u_mean.append(np.mean(slice[u2]))
        v_mean.append(np.mean(slice[v2]))
        w_mean.append(np.mean(slice[w2]))
        t_mean.append(np.mean(slice[t2]))
        rho_mean.append(np.mean(rho))
        is_temp1_fluctuating.append(slice.is_temp1_fluctuating.any())
        is_temp1_range_large.append(slice.is_temp1_range_large.any())
        is_temp2_fluctuating.append(slice.is_temp2_fluctuating.any())
        is_temp2_range_large.append(slice.is_temp2_range_large.any())

        # TODO: zrf_u, etc. NEEDS TO BE SET TO ANEM HEIGHT INITIALLY, THEN WE CAN LIN INTERP TO 10m
        try:
            blockPrint()
            coare_res = coare(Jd=jd, U=u, Zu=ZU, Tair=tair, Zt=ZT, RH=rh, Zq=ZQ, P=p, Tsea=tsea, SW_dn=sw_dn, LW_dn=LW_DN, Lat=LAT, Lon=LON, Zi=ZI, Rainrate=RAINRATE, Ts_depth=TS_DEPTH, Ss=SS, cp=None, sigH=None,zrf_u = ZU,zrf_t = ZU,zrf_q = ZU)
            enablePrint()
            tau_coare.append(coare_res[0][1])
            H_coare.append(coare_res[0][2])
        except Exception as e:
            tau_coare.append(np.nan)
            H_coare.append(np.nan)
            write_message(f"ERROR IN {fileName}: {e} - SKIPPED FOR NOW", filename='analysis_log.txt')

        # Updating time
        time_list.append(time)
        time += datetime.timedelta(minutes=TIME_INTERVAL)

        # Investigating the streak
        if tau_approx[-1]/tau_coare[-1] >= 2/0.5 and tau_approx[-1] >= 1.5:
            write_message(f"tau spike in {fileName}", filename='analysis_log.txt')
    
    if era_and_rems:
        write_message(f"Analysed {fileName} with REMS", filename='analysis_log.txt')
    else:
        write_message(f"Analysed {fileName} with ERA5", filename='analysis_log.txt')

    return (tau_approx, tau_coare, C_d, U_10_mag, H_approx, H_coare, w_turb_list, time_list, 
            u_mean, v_mean, w_mean, t_mean, rho_mean, is_temp1_fluctuating, is_temp1_range_large,
            is_temp2_fluctuating, is_temp2_range_large)

def get_time_slices(df: pd.DataFrame, interval_min: float) -> list:
    """
    Breaks df up into interval_min minute long intervals which are compiled in a list
    """
    try:
        df.Minute
    except:
        raise ValueError("This dataframe doesn't carry minute info, and is hence incompatible with get_time_slices")

    amount_of_slices = df.Minute[len(df) - 1]//interval_min
    slices = []
    for i in range(amount_of_slices):
        # Only applying a nonstrict inequality when it is at the end to stop overlap of endpoints. May cause length issues??
        if i != amount_of_slices - 1:
            slices.append(df.loc[(i*interval_min <= df.Minute) & (df.Minute < (i + 1)*interval_min)].copy(deep=True).reset_index())
        else:
            slices.append(df.loc[(i*interval_min <= df.Minute) & (df.Minute <= (i + 1)*interval_min)].copy(deep=True).reset_index())
    
    return slices

def get_turbulent(s: pd.Series) -> pd.Series:
    """
    Gets the turbulent component of entry over the total timeframe of the series s
    """
    s_bar = s.mean()
    return s - s_bar

def get_covariance(u: np.ndarray, v: np.ndarray) -> float:
    '''
    Calculates the covariances for two variables u and v.

    :param u: (np.ndarray) Var 1.
    :param v: (np.ndarray) Var 2.
    :return: (float) cov(u, v)
    '''
    #u_star_2 = np.mean(-U2_turb*w2_turb)
    #return np.mean(u*v) - np.mean(u)*np.mean(v)
    return np.cov(u, v)[0][1]

def preprocess(eraDf: pd.DataFrame, remsDf: pd.DataFrame, writeDir: os.PathLike, era_only: bool, save_plots=True, time_lim=None) -> pd.DataFrame:
    '''
    Runs any analysis/plotting prior to running it through COARE/EC/etc.

    :param eraDf: (pd.Dataframe) Df containing data from ERA5.
    :param remsDf: (pd.Dataframe) Df containing data from REMS.
    :param writeDir: (os.PathLike) Path to the save location for images.
    :param era_only: (bool) True if no REMS is being used (only ERA5).
    :param save_plots: (bool) Save if True. Otherwise plot.
    :param time_lim: (list[datetime.datetime]) A list of the form [t_start, t_end] which contains the time bounds to plot between.
    :return: (pd.DataFrame) updated eraDf and remsDf.
    '''
    # Grabbing cropped data
    if time_lim is not None:
        eraDf = eraDf[(eraDf.timemet >= time_lim[0]) & (eraDf.timemet <= time_lim[1])].reset_index(drop=True)
        remsDf = remsDf[(remsDf.timemet >= time_lim[0]) & (remsDf.timemet <= time_lim[1])].reset_index(drop=True)

    # Removing REMS xlim if it gets cropped out by time_lim
    if len(remsDf) == 0:
        era_only = True

    sns.lineplot(data=remsDf, x='timemet', y='press', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='press', label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Pressure (mBar)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'air_pressure.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='ta', label='REMS (28m AMSL)')
    sns.lineplot(data=eraDf, x='timemet', y='ta', label='ERA5 (2m AMSL)')
    plt.xlabel('time')
    plt.ylabel('Air Temperature (degC)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'air_temp.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='tsea', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='tsea', label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Sea Surface Temperature (degC)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'sea_surface_temp.png'))
        plt.close()
    else:
        plt.show()    

    # trapezoidal integration doesn't work in jupyter notebook for some reason, so we only plot in Monarch.
    if save_plots:
        time_j = []
        solrad_j = []
        time_delta = remsDf.timemet[len(remsDf) - 1] - remsDf.timemet[0]
        amountOfSlices = time_delta.total_seconds()//3600 # seconds -> hours
        solSlices = np.array_split(remsDf, amountOfSlices)
        # Integrating over each hour
        for slice in solSlices:
            slice = slice.reset_index()
            xVals = slice.timemet - slice.timemet[0]
            xVals = xVals.apply(lambda x: x.total_seconds()).values
            solrad_j.append(integrate.trapezoid(slice.solrad, x=xVals))
            time_j.append(slice.timemet[len(slice) - 1])

        time_j = []
        thermrad_j = []
        time_delta = remsDf.timemet[len(remsDf) - 1] - remsDf.timemet[0]
        amountOfSlices = time_delta.total_seconds()//3600 # seconds -> hours
        solSlices = np.array_split(remsDf, amountOfSlices)
        # Integrating over each hour
        for slice in solSlices:
            slice = slice.reset_index()
            xVals = slice.timemet - slice.timemet[0]
            xVals = xVals.apply(lambda x: x.total_seconds()).values
            thermrad_j.append(integrate.trapezoid(370*np.ones((len(xVals))), x=xVals)) # NOTE: 370 is default downward IR val
            time_j.append(slice.timemet[len(slice) - 1])

        sns.lineplot(x=time_j, y=solrad_j, markers=True, label='REMS')
        sns.lineplot(data=eraDf, x='timemet', y='solrad', markers=True, label='ERA5')
        plt.xlabel('time')
        plt.ylabel('Downward Solar Radiation (J/m^2)')
        if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
        plt.xticks(plt.xticks()[0], rotation=90)
        if save_plots:
            plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'downward_solar_rad_int.png'))
            plt.close()
        else:
            plt.show()

        sns.lineplot(x=time_j, y=thermrad_j, markers=True, label='REMS')
        sns.lineplot(data=eraDf, x='timemet', y='thermrad', markers=True, label='ERA5')
        plt.xlabel('time')
        plt.ylabel('Downward IR Radiation (J/m^2)')
        if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
        plt.xticks(plt.xticks()[0], rotation=90)
        if save_plots:
            plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'downward_IR_rad_int.png'))
            plt.close()
        else:
            plt.show()    

        # TODO: PATCH FIX
        eraDf.solrad = eraDf.solrad/3600
        eraDf.thermrad = eraDf.thermrad/3600

    sns.lineplot(data=remsDf, x='timemet', y='solrad', markers=True, label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='solrad', markers=True, label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Downward Solar Radiation (W/m^2)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'downward_solar_diff.png'))
        plt.close()
    else:
        plt.show()    

    # NOTE: Missing plots: water current speeds

    sns.lineplot(data=remsDf, x='timemet', y='rh', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='rh', label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Relative humidity (%)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'rel_hum.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='spech', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='spech', label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Specific humidity (kg/kg)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'spec_hum.png'))
        plt.close()
    else:
        plt.show()    

    return eraDf, remsDf

def postprocess(outDf: pd.DataFrame, eraDf: pd.DataFrame, remsDf: pd.DataFrame, writeDir: os.PathLike, era_only: bool, save_plots=True, time_lim=None) -> None:
    '''
    Runs all the plotting and postprocessing after data generation from COARE/EC/etc. is complete.

    :param outDf: (pd.Dataframe) Df containing the outputs from analysis_loop.
    :param eraDf: (pd.Dataframe) Df containing data from ERA5.
    :param remsDf: (pd.Dataframe) Df containing data from REMS.
    :writeDir: (os.PathLike) Path to the save location for images.
    :param era_only: (bool) True if no REMS is being used (only ERA5).
    :param save_plots: (bool) Save if True. Otherwise plot.
    :param time_lim: (list[datetime.datetime]) A list of the form [t_start, t_end] which contains the time bounds to plot between.
    '''
    # Grabbing cropped data
    if time_lim is not None:
        outDf = outDf[(outDf.time >= time_lim[0]) & (outDf.time <= time_lim[1])].reset_index(drop=True)
        eraDf = eraDf[(eraDf.timemet >= time_lim[0]) & (eraDf.timemet <= time_lim[1])].reset_index(drop=True)
        remsDf = remsDf[(remsDf.timemet >= time_lim[0]) & (remsDf.timemet <= time_lim[1])].reset_index(drop=True)

    # Removing REMS xlim if it gets cropped out by time_lim
    if len(remsDf) == 0:
        era_only = True

    sns.lineplot(data=outDf, x='time', y='rho', markers=True)
    plt.xlabel('time')
    plt.ylabel('Air Density (kg/m^3)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'air_dens.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='u', label="Anem U Component")
    sns.lineplot(data=eraDf, x='timemet', y='v_10', label="ERA5 V Component (10m)")
    plt.xlabel('time')
    plt.ylabel('Northerly Component of Wind Speed (m/s)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'north_wind.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='v', label="Anem V Component")
    sns.lineplot(data=eraDf, x='timemet', y='u_10', label="ERA5 U Component (10m)")
    plt.xlabel('time')
    plt.ylabel('Easterly Component of Wind Speed (m/s)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'east_wind.png'))
        plt.close()
    else:
        plt.show()

    sns.lineplot(data=outDf, x='time', y='w', label="Anem W Component")
    plt.xlabel('time')
    plt.ylabel('Upward Component of Wind Speed (m/s)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'upward_wind.png'))
        plt.close()
    else:
        plt.show()

    sns.lineplot(data=outDf, x='time', y='U_10', label='Anem')
    sns.lineplot(x=eraDf.timemet, y=np.linalg.norm(eraDf[['v_10', 'u_10']].values,axis=1), label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Magnitude of Horizontal Wind Speed at 10m (m/s)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'east_wind.png'))
        plt.close()
    else:
        plt.show()

    sns.lineplot(data=outDf, x='time', y='ta', label="Anem")
    sns.lineplot(data=eraDf, x='timemet', y='ta', label="ERA5")
    plt.xlabel('time')
    plt.ylabel('Sea Surface Temperature (degC)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'sea_surf_temp.png'))
        plt.close()
    else:
        plt.show()   

    lin_lims = [min([min(outDf.tauCoare), min(outDf.tauApprox)]), max([max(outDf.tauCoare), max(outDf.tauApprox)])]
    sns.regplot(data=outDf, x='tauCoare', y='tauApprox', label='Best fit with 95% CI')
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Shear Stress')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_xy.png'))
        plt.close()
    else:
        plt.show()   

    lin_lims = [min([min(outDf.HCoare), min(outDf.HApprox)]), max([max(outDf.HCoare), max(outDf.HApprox)])]
    sns.regplot(data=outDf, x='HCoare', y='HApprox', label='Best fit with 95% CI')
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Sensible Heat Flux')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_xy.png'))
        plt.close()
    else:
        plt.show()   

    outDf.Cd = 1000*outDf.Cd

    sns.scatterplot(data=outDf, x='U_10', y='Cd')
    # plt.xlim([0, 25])
    # plt.ylim([-2,5])
    plt.xlabel('U_10 (m/s)')
    plt.ylabel('1000*Cd')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'Cd_spread.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='tauApprox', label="EC", markers=True)
    sns.lineplot(data=outDf, x='time', y='tauCoare', label="COARE", markers=True)
    sns.lineplot(x=outDf.time, y=outDf.tauApprox.rolling(window=5, step=5).mean())
    plt.xlabel('time')
    plt.ylabel('Shear Stress')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_timeseries.png'))
        plt.close()
    else:
        plt.show()  

    sns.lineplot(data=outDf, x='time', y='HApprox', label="EC", markers=True)
    sns.lineplot(data=outDf, x='time', y='HCoare', label="COARE", markers=True)
    sns.lineplot(x=outDf.time, y=outDf.HApprox.rolling(window=5, step=5).mean())
    plt.xlabel('time')
    plt.ylabel('Sensible Heat Flux')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show()     

    ax1 = sns.lineplot(data=outDf, x='time', y='HApprox', label="EC Sensible Heat Flux", legend=False)
    ax2 = plt.twinx()
    sns.lineplot(data=eraDf, x='timemet', y='crr', label='Convective Rain Rate (kg m^-2 s^-1)', ax=ax2, color='orange', legend=False)
    ax1.figure.legend()
    plt.xlabel('time')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show() 

    ax1 = sns.lineplot(data=outDf, x='time', y='HApprox', label="EC Sensible Heat Flux", legend=False)
    ax2 = plt.twinx()
    sns.lineplot(data=eraDf, x='timemet', y='swh', label='Significant Wave Height (m)', ax=ax2, color='orange', legend=False)
    ax1.figure.legend()
    plt.xlabel('time')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show()   

    ax1 = sns.lineplot(data=outDf, x='time', y='HCoare', label="COARE Sensible Heat Flux", legend=False)
    ax2 = plt.twinx()
    sns.lineplot(data=eraDf, x='timemet', y='swh', label='Significant Wave Height (m)', ax=ax2, color='orange', legend=False)
    ax1.figure.legend()
    plt.xlabel('time')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show()

    # fig, ax = plt.subplots()
    # lns1 = ax.plot(outDf.time, outDf.HApprox, "-o", label='EC')
    # lns2 = ax.plot(outDf.time, outDf.HCoare, "-o", label='COARE')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Sensible Heat Flux')
    # ax2 = ax.twinx()
    # lns3 = ax2.plot(outDf.time, outDf.U_10, "-o", color='r', label='U_10')
    # ax2.set_ylim([0,20])
    # ax2.set_ylabel('U_10 (m/s)')
    # lns = lns1+lns2+lns3
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc=0)
    # plt.savefig(os.path.join(writeDir, f"{title}.png"))
    # plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', nargs='+', type=str, help='Path to the rawdata. Can be a list.')
    parser.add_argument('--write_dir', nargs='+', type=str, help='Path to output. Can be a list.')
    parser.add_argument('--cpu_fraction', type=float, help='% Of CPUs to use. Can be within (0,1].', default=1)
    parser.add_argument('--run_supervised', action='store_true', help='Run one-by-one analysis', default=False)
    parser.add_argument('--era_only', action='store_true', help='If True, always use ERA5 and never use REMS for relevant parameters. If False, will use REMS when available and ERA5 otherwise.', default=False)
    parser.add_argument('--no_era', action='store_true', help='If True, will never use ERA5 - only REMS (skips unavailable times).', default=False)
    args = parser.parse_args()

    # Using a modified version of np.load to read data with allow_pickle turned off in the .npz file
    np_load_modified = lambda *a,**k: np.load(*a, allow_pickle=True, **k)

    # Grabbing REMS stuff
    for cyclone in ['quang']:
        with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'REMS', f'meteo_{cyclone}.npz')) as metFile:
            timemet = metFile['timemet.npy'] # YYYYMMDD and milliseconds past midnight
            press = metFile['press.npy'] # Barometric Pressure (hPa=mbar)
            rh = metFile['rh.npy'] # Relative Humidity (%)
            spech = metFile['spech.npy'] # Specific humidity (rh: ratio, p: Pa; T: Kelvin)
            ta = metFile['ta.npy'] # Air Temperature (C)
            solrad = metFile['solrad.npy'] # Downward Solar radiation (Wm^-2)
        with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'REMS', f'meteo_{cyclone}_currents.npz')) as metFile:
            #timemet = metFile['timemet.npy'] # YYYYMMDD and milliseconds past midnight
            cur_n_comp = metFile['cur_n_comp.npy'] # Northward component of current velocity (m/s)
            cur_e_comp = metFile['cur_e_comp.npy'] # Eastward component of current velocity (m/s)
            tsea = metFile['tsea.npy'] # Water temperature (degC)
            depth = metFile['depth.npy'] # Approx. distance from surface (m), Babanin et al.

    remsDf = pd.DataFrame({"timemet": timemet, "press": press, "rh": rh, "spech": spech, "ta": ta, "solrad": solrad,
                            "cur_n_comp": cur_n_comp, "cur_e_comp": cur_e_comp, "tsea": tsea, "depth": depth})

    # Grabbing ERA5 data
    with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'ERA5', 'era2015.npz')) as eraFile:
        timemet = eraFile['timemet.npy']
        u_10 = eraFile['u_10.npy'] # 10 metre U wind component (m/s)
        v_10 = eraFile['v_10.npy'] # 10 metre V wind component (m/s)
        ta = eraFile['two_m_temp.npy'] - KELVIN_TO_CELSIUS # 2 metre air temperature (degC)
        rh = eraFile['rh.npy'] # Relative Humidity (%)
        spech = eraFile['spechum.npy'] # Specific Humidity (%)
        waveDir = eraFile['mean_wave_dir.npy'] # Mean wave direction in true deg (0deg North)
        tsea = eraFile['surface_temp.npy'] - KELVIN_TO_CELSIUS # Sea temperature near surface (degC)
        press = eraFile['surface_pres.npy']/100 # Surface pressure (mBar)
        solrad = eraFile['surface_solrad.npy'] # Surface solar radiation downwards (J/m^2)
        thermrad = eraFile['surface_thermrad.npy'] # Surface thermal radiation downwards (J/m^2)
        crr = eraFile['crr.npy'] # Precipitation rate (kgm^-2s^-1)
        swh = eraFile['swh.npy'] # Significant wave height of wind waves + swell (m) (http://www.bom.gov.au/marine/knowledge-centre/reference/waves.shtml)

    eraDf = pd.DataFrame({"timemet": timemet, "u_10": u_10, "v_10": v_10, "tsea": tsea, "waveDir": waveDir, 
                            "ta": ta, "rh": rh, "spech": spech, "press": press, "solrad": solrad, "thermrad": thermrad,
                            "crr": crr, "swh": swh})

    sns.set_theme(style='darkgrid')

    t0 = time.perf_counter()
    write_message(f"Starting Analysis Run", filename='analysis_log.txt', writemode='w')
    for i, _ in enumerate(args.read_dir):
        readDir = Path(args.read_dir[i])
        writeDir = Path(args.write_dir[i])

        # Making folders
        os.mkdir(os.path.join(writeDir, 'Preprocess'))
        os.mkdir(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA'))
        os.mkdir(os.path.join(writeDir, 'Postprocess'))

        eraDf, remsDf = preprocess(eraDf, remsDf, writeDir=writeDir, era_only=args.era_only)
        outDf = analysis_loop(readDir, eraDf, remsDf, supervised=args.run_supervised, cpuFraction=args.cpu_fraction, era_only=args.era_only, no_era=args.no_era)

        outDf = outDf.sort_values(by='time') # Sorting outDf since it may be jumbled due to multiprocessing

        postprocess(outDf, eraDf, remsDf, writeDir=writeDir, era_only=args.era_only)

        eraDf.to_csv(os.path.join(writeDir, 'eraDf.csv'))
        remsDf.to_csv(os.path.join(writeDir, 'remsDf.csv'))
        outDf.to_csv(os.path.join(writeDir, 'outDf.csv'))

    t1 = time.perf_counter()
    
    write_message(f"Took {(t1-t0)/60}min", filename='analysis_log.txt')