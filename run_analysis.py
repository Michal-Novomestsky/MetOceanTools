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
from Modules.DataCleaner import apply_window_wise
from COARE.COARE3_6.coare36vnWarm_et import coare36vnWarm_et as coare

# Defining constants
KELVIN_TO_CELSIUS = 273.15
ZU = 14.8 # Height of anemometer #1 (the higher one)
ZT = 28 # Approx. height of flare bridge AMSL
ZQ = 28 # Approx. height of flare bridge AMSL
LAT = -19.5856 # 19.5856S (Babanin et al.)
LON = 116.1367 # 116.1367E
SS = 35 # https://salinity.oceansciences.org/overview.htm
CPD = hum.cpd # Isobaric specific heat of dry air at constant pressure [J/(kg K)]
TIME_INTERVAL = 10
WINDOW_WIDTH = 5 # Amount of datapoints to consider at a time when averaging for plots
ANEM1_TO_U10 = (10/ZU)**0.11 # Extrapolation scale factor

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
    collector_u1 = []
    collector_u1_turb = []
    collector_v1 = []
    collector_v1_turb = []
    collector_w1 = []
    collector_w1_turb = []
    collector_t1 = []
    collector_u2 = []
    collector_u2_turb = []
    collector_v2 = []
    collector_v2_turb = []
    collector_w2 = []
    collector_w2_turb = []
    collector_t2 = []
    collector_rho = []
    collector_t1_fluct = []
    collector_t1_rng = []
    collector_t2_fluct = []
    collector_t2_rng = []
    collector_u_star_1 = []

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
                collector_time += output[6]
                collector_u1 += output[7]
                collector_u1_turb += output[8]
                collector_v1 += output[9]
                collector_v1_turb += output[10]
                collector_w1 += output[11]
                collector_w1_turb += output[12]
                collector_t1 += output[13]
                collector_u2 += output[14]
                collector_u2_turb += output[15]
                collector_v2 += output[16]
                collector_v2_turb += output[17]
                collector_w2 += output[18]
                collector_w2_turb += output[19]
                collector_t2 += output[20]
                collector_rho += output[21]
                collector_t1_fluct += output[22]
                collector_t1_rng += output[23]
                collector_t2_fluct += output[24]
                collector_t2_rng += output[25]
                collector_u_star_1 += output[26]

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
                    collector_time += outputElem[6]
                    collector_u1 += outputElem[7]
                    collector_u1_turb += outputElem[8]
                    collector_v1 += outputElem[9]
                    collector_v1_turb += outputElem[10]
                    collector_w1 += outputElem[11]
                    collector_w1_turb += outputElem[12]
                    collector_t1 += outputElem[13]
                    collector_u2 += outputElem[14]
                    collector_u2_turb += outputElem[15]
                    collector_v2 += outputElem[16]
                    collector_v2_turb += outputElem[17]
                    collector_w2 += outputElem[18]
                    collector_w2_turb += outputElem[19]
                    collector_t2 += outputElem[20]
                    collector_rho += outputElem[21]
                    collector_t1_fluct += outputElem[22]
                    collector_t1_rng += outputElem[23]
                    collector_t2_fluct += outputElem[24]
                    collector_t2_rng += outputElem[25]
                    collector_u_star_1 += outputElem[26]

    write_message("Analysis run done!", filename='analysis_log.txt')
    return pd.DataFrame({"time": collector_time, "tauApprox": collector_tauApprox, "tauCoare": collector_tauCoare,
                            "Cd": collector_Cd, "U_10": collector_U_10, "HApprox": collector_HApprox, "HCoare": collector_HCoare, 
                            "u1": collector_u1, "u1_turb": collector_u1_turb, "v1": collector_v1, "v1_turb": collector_v1_turb, "w1": collector_w1, "w1_turb": collector_w1_turb,
                            "ta_1": collector_t1, "u2": collector_u2, "u2_turb": collector_u2_turb, "v2": collector_v2, "v2_turb": collector_v2_turb, "w2": collector_w2, "w2_turb": collector_w2_turb, 
                            "ta_2": collector_t2, "rho": collector_rho, "is_temp1_fluctuating": collector_t1_fluct, "is_temp1_range_large": collector_t1_rng, 
                            "is_temp2_fluctuating": collector_t2_fluct, "is_temp2_range_large": collector_t2_rng, "u_star_1": collector_u_star_1})

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
    U_10_mean = [] # NOTE: "_mag" is to prevent it being const from all caps
    u_star_1_list = []
    u1_mean = []
    u1_turb_mean = []
    v1_mean = []
    v1_turb_mean = []
    w1_mean = []
    w1_turb_mean = []
    t1_mean = []
    u2_mean = []
    u2_turb_mean = []
    v2_mean = []
    v2_turb_mean = []
    w2_mean = []
    w2_turb_mean = []
    t2_mean = []
    rho_mean = []
    is_temp1_fluctuating = []
    is_temp1_range_large = []
    is_temp2_fluctuating = []
    is_temp2_range_large = []

    u1 = "Anemometer #1 U Velocity (ms-1)"
    v1 = "Anemometer #1 V Velocity (ms-1)"
    w1 = "Anemometer #1 W Velocity (ms-1)"
    t1 = "Anemometer #1 Temperature (degC)"
    comp1 = "Compass #1 (deg)"
    u2 = "Anemometer #2 U Velocity (ms-1)"
    v2 = "Anemometer #2 V Velocity (ms-1)"
    w2 = "Anemometer #2 W Velocity (ms-1)"
    t2 = "Anemometer #2 Temperature (degC)"
    comp2 = "Compass #2 (deg)"

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
        slice = slice[~slice.is_temp1_range_large] # Removing erroneous points
        slice[u2] = -slice[u2]
        slice[v2] = -slice[v2]

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

        # DERIVED FROM ANEM 1 (MRU CORRECTED ONE)
        U_10_vec, U_anem1_mag, U_10_turb, w_turb, T_turb = get_windspeed_data(slice, u1, v1, w1, t1)
        # U_10_mag = ANEM1_TO_U10*U_anem1_mag
        U_10_mag = U_anem1_mag

        # u_AirWat = u_Air - u_Wat
        #U_vec.East = U_vec.East - remsSlice.cur_e_comp # Seem to be negligible compared to wind speed
        #U_vec.North = U_vec.North - remsSlice.cur_n_comp
        # u = np.sqrt(U_10_vec.North**2 + U_10_vec.East**2) #TODO CHANGE TO U_10_mag

        u_star_1 = get_covariance(U_10_turb, w_turb)
        tau_approx.append(-rho*u_star_1)
        H_approx.append(rho*CPD*get_covariance(w_turb, T_turb))

        #TODO: Assume U_10 ~= U_14.8 for now
        # C_d.append(np.mean(-U_10_turb*w_turb)/(np.mean(U_10_mag)**2))
        C_d.append(-u_star_1/(np.mean(U_10_mag)**2))
        u_star_1_list.append(u_star_1)
        U_10_mean.append(np.mean(U_10_mag))
        u1_mean.append(np.mean(slice[u1]))
        u1_turb_mean.append(np.mean(get_turbulent(slice[u1])))
        v1_mean.append(np.mean(slice[v1]))
        v1_turb_mean.append(np.mean(get_turbulent(slice[v1])))
        w1_mean.append(np.mean(slice[w1]))
        w1_turb_mean.append(np.mean(get_turbulent(slice[w1])))
        t1_mean.append(np.mean(slice[t1]))
        u2_mean.append(np.mean(slice[u2]))
        u2_turb_mean.append(np.mean(get_turbulent(slice[u2])))
        v2_mean.append(np.mean(slice[v2]))
        v2_turb_mean.append(np.mean(get_turbulent(slice[v2])))
        w2_mean.append(np.mean(slice[w2]))
        w2_turb_mean.append(np.mean(get_turbulent(slice[w2])))
        t2_mean.append(np.mean(slice[t2]))
        rho_mean.append(np.mean(rho))
        is_temp1_fluctuating.append(slice.is_temp1_fluctuating.any())
        is_temp1_range_large.append(slice.is_temp1_range_large.any())
        is_temp2_fluctuating.append(slice.is_temp2_fluctuating.any())
        is_temp2_range_large.append(slice.is_temp2_range_large.any())

        # TODO: zrf_u, etc. NEEDS TO BE SET TO ANEM HEIGHT INITIALLY, THEN WE CAN LIN INTERP TO 10m
        try:
            blockPrint()
            coare_res = coare(Jd=jd, U=np.mean(U_10_mag), Zu=ZU, Tair=tair, Zt=ZT, RH=rh, Zq=ZQ, P=p, 
                              Tsea=tsea, SW_dn=sw_dn, LW_dn=LW_DN, Lat=LAT, Lon=LON, Zi=ZI, 
                              Rainrate=RAINRATE, Ts_depth=TS_DEPTH, Ss=SS, cp=None, sigH=None,
                              zrf_u=ZU, zrf_t=ZU, zrf_q=ZU)
            enablePrint()
            tau_coare.append(coare_res[0][1])
            H_coare.append(coare_res[0][2])
        except Exception as e:
            # tau_coare.append(-0.3)
            # H_coare.append(-10)
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

    return (tau_approx, tau_coare, C_d, U_10_mean, H_approx, H_coare, time_list, 
            u1_mean, u1_turb_mean, v1_mean, v1_turb_mean, w1_mean, w1_turb_mean, t1_mean, 
            u2_mean, u2_turb_mean, v2_mean, v2_turb_mean, w2_mean, w2_turb_mean, 
            t2_mean, rho_mean, is_temp1_fluctuating, is_temp1_range_large,
            is_temp2_fluctuating, is_temp2_range_large, u_star_1_list)

def get_windspeed_data(slice: pd.Series, u: str, v: str, w: str, t: str) -> tuple:
    w_turb = get_turbulent(slice[w])
    T_turb = get_turbulent(slice[t])
    #T_turb = T_turb/(1 + 0.378*e/p)

    # Getting magnitude of turbulent horizontal velocity vector
    U_turb = get_turbulent(np.sqrt(slice[u]**2 + slice[v]**2))

    # Getting current-corrected windspeed
    U_mag = np.sqrt(slice[u]**2 + slice[v]**2)
    # Easterly -> +ive x axis, Northerly -> +ive y. Note that anem v+ is west so east is -v
    U_vec = pd.DataFrame({'East': slice[v], 'North': slice[u]})
    U_vec = U_vec.mean() #Taking TIME_INTERVAL min avg

    return U_vec, U_mag, U_turb, w_turb, T_turb

def get_time_slices(df: pd.DataFrame, interval_min: float) -> list:
    """
    Breaks df up into interval_min minute long intervals which are compiled in a list
    """
    # try:
    #     df.Minute
    # except:
    #     raise ValueError("This dataframe doesn't carry minute info, and is hence incompatible with get_time_slices")

    # amount_of_slices = df.Minute[len(df) - 1]//interval_min
    # slices = []
    # for i in range(amount_of_slices):
    #     # Only applying a nonstrict inequality when it is at the end to stop overlap of endpoints. May cause length issues??
    #     if i != amount_of_slices - 1:
    #         slices.append(df.loc[(i*interval_min <= df.Minute) & (df.Minute < (i + 1)*interval_min)].copy(deep=True).reset_index())
    #     else:
    #         slices.append(df.loc[(i*interval_min <= df.Minute) & (df.Minute <= (i + 1)*interval_min)].copy(deep=True).reset_index())
    print(pd.isna(df))
    print(pd.isna(df).any())
    if pd.isna(df).any():
        print(df)
        print(df[pd.isna(df)])

    window_width = round((interval_min*60)/(df.GlobalSecs[1] - df.GlobalSecs[0])) # Amount of indicies to consider = wanted_stepsize/data_stepsize
    slices = df.rolling(window=window_width, step=window_width)

    return [slice for slice in slices]

def get_turbulent(s: pd.Series) -> pd.Series:
    """
    Gets the turbulent component of entry over the total timeframe of the series s
    """
    s_bar = s.mean()
    return s - s_bar

def get_covariance(u: np.ndarray, v: np.ndarray) -> float:
    '''
    Calculates the covariance for two variables u and v.

    :param u: (np.ndarray) Var 1.
    :param v: (np.ndarray) Var 2.
    :return: (float) cov(u, v)
    '''
    logical = ((pd.notna(u)) & (pd.notna(v)))
    u = u[logical]
    v = v[logical]
    if len(u) <= 1 or len(v) <= 1:
        return np.nan
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
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Pressure (mBar)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'air_pressure.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='ta', label='REMS (28m AMSL)')
    sns.lineplot(data=eraDf, x='timemet', y='ta', label='ERA5 (2m AMSL)')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Air Temperature (degC)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'air_temp.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='tsea', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='tsea', label='ERA5')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Sea Surface Temperature (degC)')
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
        if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
        plt.xlabel('time')
        plt.ylabel('Downward Solar Radiation (J/m^2)')
        plt.xticks(plt.xticks()[0], rotation=90)
        if save_plots:
            plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'downward_solar_rad_int.png'))
            plt.close()
        else:
            plt.show()

        sns.lineplot(x=time_j, y=thermrad_j, markers=True, label='REMS')
        sns.lineplot(data=eraDf, x='timemet', y='thermrad', markers=True, label='ERA5')
        if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
        plt.xlabel('time')
        plt.ylabel('Downward IR Radiation (J/m^2)')
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
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Downward Solar Radiation (W/m^2)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'downward_solar_diff.png'))
        plt.close()
    else:
        plt.show()    

    # NOTE: Missing plots: water current speeds

    sns.lineplot(data=remsDf, x='timemet', y='rh', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='rh', label='ERA5')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Relative humidity (%)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA', 'rel_hum.png'))
        plt.close()
    else:
        plt.show()    

    sns.lineplot(data=remsDf, x='timemet', y='spech', label='REMS')
    sns.lineplot(data=eraDf, x='timemet', y='spech', label='ERA5')
    if not era_only: plt.xlim([remsDf.timemet[0], remsDf.timemet[len(remsDf) - 1]])
    plt.xlabel('time')
    plt.ylabel('Specific humidity (kg/kg)')
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
    if not era_only:
        outDf = outDf[(outDf.time >= remsDf.timemet[0]) & (outDf.time <= remsDf.timemet[len(remsDf) - 1])].reset_index(drop=True)
        eraDf = eraDf[(eraDf.timemet >= remsDf.timemet[0]) & (eraDf.timemet <= remsDf.timemet[len(remsDf) - 1])].reset_index(drop=True)


    # Removing REMS xlim if it gets cropped out by time_lim
    if len(remsDf) == 0:
        era_only = True

    sns.lineplot(data=outDf, x='time', y='rho', markers=True)
    plt.xlabel('time')
    plt.ylabel('Air Density (kg/m^3)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'air_dens.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='u1', label="Anem 1 U Component")
    sns.lineplot(data=outDf, x='time', y='u2', label="Anem 2 U Component")
    sns.lineplot(data=eraDf, x='timemet', y='v_10', label="ERA5 V Component (10m)")
    plt.xlabel('time')
    plt.ylabel('Northerly Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'north_wind.png'))
        plt.close()
    else:
        plt.show() 

    mean_u1_turb = apply_window_wise(outDf.u1_turb, WINDOW_WIDTH, np.mean)
    mean_u2_turb = apply_window_wise(outDf.u2_turb, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='u1_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 1 U Turbulent Component")
    sns.scatterplot(data=outDf, x='time', y='u2_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 2 U Turbulent Component")
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_u1_turb, label='Anem 1')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_u2_turb, label='Anem 2')
    plt.xlabel('time')
    plt.ylabel('Northerly Turbulent Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='v1', label="Anem 1 V Component")
    sns.lineplot(data=outDf, x='time', y='v2', label="Anem 2 V Component")
    sns.lineplot(data=eraDf, x='timemet', y='u_10', label="ERA5 U Component (10m)")
    plt.xlabel('time')
    plt.ylabel('Easterly Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'east_wind.png'))
        plt.close()
    else:
        plt.show()

    mean_v1_turb = apply_window_wise(outDf.v1_turb, WINDOW_WIDTH, np.mean)
    mean_v2_turb = apply_window_wise(outDf.v2_turb, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='v1_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 1 V Turbulent Component")
    sns.scatterplot(data=outDf, x='time', y='v2_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 2 V Turbulent Component")
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_v1_turb, label='Anem 1')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_v2_turb, label='Anem 2')
    plt.xlabel('time')
    plt.ylabel('Easterly Turbulent Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='w1', label="Anem 1 W Component")
    sns.lineplot(data=outDf, x='time', y='w2', label="Anem 2 W Component")
    plt.xlabel('time')
    plt.ylabel('Upward Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'upward_wind.png'))
        plt.close()
    else:
        plt.show()

    mean_w1_turb = apply_window_wise(outDf.w1_turb, WINDOW_WIDTH, np.mean)
    mean_w2_turb = apply_window_wise(outDf.w2_turb, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='w1_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 1 W Turbulent Component")
    sns.scatterplot(data=outDf, x='time', y='w2_turb', marker='.', label=f"{TIME_INTERVAL}min Avg Anem 2 W Turbulent Component")
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_w1_turb, label='Anem 1')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_w2_turb, label='Anem 2')
    plt.xlabel('time')
    plt.ylabel('Upward Turbulent Component of Wind Speed (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='U_10', label='Anem')
    sns.lineplot(x=eraDf.timemet, y=np.linalg.norm(eraDf[['v_10', 'u_10']].values,axis=1), label='ERA5')
    plt.xlabel('time')
    plt.ylabel('Magnitude of Horizontal Wind Speed at 10m (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'east_wind.png'))
        plt.close()
    else:
        plt.show()

    sns.lineplot(data=outDf, x='time', y='ta_1', label="Anem 1")
    sns.lineplot(data=outDf, x='time', y='ta_2', label="Anem 2")
    sns.lineplot(data=eraDf, x='timemet', y='ta', label="ERA5")
    plt.xlabel('time')
    plt.ylabel('Air Temperature (degC)')
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

    lin_lims = [min([min(outDf.tauCoare), min(outDf.tauApprox)]), max([max(outDf.tauCoare), max(outDf.tauApprox)])]
    sns.kdeplot(data=outDf, x='tauCoare', y='tauApprox', fill=True, levels=100, cmap='mako', thresh=0)
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

    lin_lims = [min([min(outDf.HCoare), min(outDf.HApprox)]), max([max(outDf.HCoare), max(outDf.HApprox)])]
    sns.kdeplot(data=outDf, x='HCoare', y='HApprox', fill=True, levels=100, cmap='mako', thresh=0)
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Sensible Heat Flux')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_xy.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='u_star_1', label="Anem 1 u_star", marker='.')
    plt.xlabel('time')
    plt.ylabel('u_star (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    plt.show()

    # Making a box for x [0, 25], y [-2, 5]
    left_wall = [[0, 0], [-2, 5]]
    right_wall = [[25, 25], [-2, 5]]
    bottom_wall = [[0, 25], [-2, -2]]
    top_wall = [[0, 25], [5, 5]]
    sns.lineplot(x=left_wall[0], y=left_wall[1], color='black')
    sns.lineplot(x=right_wall[0], y=right_wall[1], color='black')
    sns.lineplot(x=bottom_wall[0], y=bottom_wall[1], color='black')
    sns.lineplot(x=top_wall[0], y=top_wall[1], color='black')
    sns.scatterplot(x=outDf.U_10, y=1000*outDf.Cd)
    plt.xlabel('U_10 (m/s)')
    plt.ylabel('1000*Cd')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'Cd_spread.png'))
        plt.close()
    else:
        plt.show()   

    mean_ec = apply_window_wise(outDf.tauApprox, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='tauApprox', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='tauCoare', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Shear Stress')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_timeseries.png'))
        plt.close()
    else:
        plt.show()  

    mean_ec = apply_window_wise(outDf.HApprox, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='HApprox', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='HCoare', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Sensible Heat Flux')
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
    
    write_message(f"Took {round((t1-t0)/60, 1)}min", filename='analysis_log.txt')