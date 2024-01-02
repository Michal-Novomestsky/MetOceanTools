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
ZU_1 = 14.8 # Height of anemometer #1 (MRU available)
ZU_2 = 8.8 # Height of anemometer #2 (no MRU)
ZT = 28 # Approx. height of flare bridge AMSL
ZQ = 28 # Approx. height of flare bridge AMSL
LASER_TO_ANEM_1 = ZT - ZU_1
LASER_TO_ANEM_2 = ZT - ZU_2
LAT = -19.5856 # 19.5856S (Babanin et al.)
LON = 116.1367 # 116.1367E
SS = 35 # https://salinity.oceansciences.org/overview.htm
CPD = hum.cpd # Isobaric specific heat of dry air at constant pressure [J/(kg K)]
TIME_INTERVAL = 10
MIN_COV_SIZE = 0.99 # Minimum % of points retained for valid covariance calculation
MIN_SLICE_SIZE = 1000 # Minimum slice size prior to chopping out data
WINDOW_WIDTH = 5 # Amount of datapoints to consider at a time when averaging for plots
# ANEM1_TO_U10 = (10/ZU_1)**0.11 # Extrapolation scale factor
# ANEM2_TO_U10 = (10/ZU_2)**0.11

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
    collector_rho = []
    collector_t1_fluct = []
    collector_t1_rng = []
    collector_t2_fluct = []
    collector_t2_rng = []
    collector_laser1 = []
    collector_laser2 = []
    collector_laser3 = []
    collector_laser4 = []

    collector_tauApprox_1 = []
    collector_tauCoare_1 = []
    collector_HApprox_1 = []
    collector_HCoare_1 = []
    collector_Cd_1 = []
    collector_Cd_coare_1 = []
    collector_u_star_1 = []
    collector_U_anem_1 = []
    collector_u1 = []
    collector_u1_turb = []
    collector_v1 = []
    collector_v1_turb = []
    collector_w1 = []
    collector_w1_turb = []
    collector_t1 = []
    collector_zu1 = []

    collector_tauApprox_2 = []
    collector_tauCoare_2 = []
    collector_HApprox_2 = []
    collector_HCoare_2 = []
    collector_Cd_2 = []
    collector_Cd_coare_2 = []
    collector_u_star_2 = []
    collector_U_anem_2 = []
    collector_u2 = []
    collector_u2_turb = []
    collector_v2 = []
    collector_v2_turb = []
    collector_w2 = []
    collector_w2_turb = []
    collector_t2 = []
    collector_zu2 = []

    # One-by-one
    if supervised:
        for file in files:
            output = _analysis_iteration(file, eraDf, remsDf, era_only, no_era)
            if output is not None:
                collector_tauApprox_1 += output[0]
                collector_tauCoare_1 += output[1]
                collector_Cd_1 += output[2]
                collector_U_anem_1 += output[3]
                collector_HApprox_1 += output[4]
                collector_HCoare_1 += output[5]
                collector_time += output[6]
                collector_u1 += output[7]
                collector_u1_turb += output[8]
                collector_v1 += output[9]
                collector_v1_turb += output[10]
                collector_w1 += output[11]
                collector_w1_turb += output[12]
                collector_t1 += output[13]
                collector_tauApprox_2 += output[14]
                collector_tauCoare_2 += output[15]
                collector_HApprox_2 += output[16]
                collector_HCoare_2 += output[17]
                collector_Cd_2 += output[18]
                collector_U_anem_2 += output[19]
                collector_u_star_2 += output[20]
                collector_u2 += output[21]
                collector_u2_turb += output[22]
                collector_v2 += output[23]
                collector_v2_turb += output[24]
                collector_w2 += output[25]
                collector_w2_turb += output[26]
                collector_t2 += output[27]
                collector_rho += output[28]
                collector_t1_fluct += output[29]
                collector_t1_rng += output[30]
                collector_t2_fluct += output[31]
                collector_t2_rng += output[32]
                collector_u_star_1 += output[33]
                collector_Cd_coare_1 += output[34]
                collector_Cd_coare_2 += output[35]
                collector_zu1 += output[36]
                collector_zu2 += output[37]
                collector_laser1 += output[38]
                collector_laser2 += output[39]
                collector_laser3 += output[40]
                collector_laser4 += output[41]

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
                    collector_tauApprox_1 += outputElem[0]
                    collector_tauCoare_1 += outputElem[1]
                    collector_Cd_1 += outputElem[2]
                    collector_U_anem_1 += outputElem[3]
                    collector_HApprox_1 += outputElem[4]
                    collector_HCoare_1 += outputElem[5]
                    collector_time += outputElem[6]
                    collector_u1 += outputElem[7]
                    collector_u1_turb += outputElem[8]
                    collector_v1 += outputElem[9]
                    collector_v1_turb += outputElem[10]
                    collector_w1 += outputElem[11]
                    collector_w1_turb += outputElem[12]
                    collector_t1 += outputElem[13]
                    collector_tauApprox_2 += outputElem[14]
                    collector_tauCoare_2 += outputElem[15]
                    collector_HApprox_2 += outputElem[16]
                    collector_HCoare_2 += outputElem[17]
                    collector_Cd_2 += outputElem[18]
                    collector_U_anem_2 += outputElem[19]
                    collector_u_star_2 += outputElem[20]
                    collector_u2 += outputElem[21]
                    collector_u2_turb += outputElem[22]
                    collector_v2 += outputElem[23]
                    collector_v2_turb += outputElem[24]
                    collector_w2 += outputElem[25]
                    collector_w2_turb += outputElem[26]
                    collector_t2 += outputElem[27]
                    collector_rho += outputElem[28]
                    collector_t1_fluct += outputElem[29]
                    collector_t1_rng += outputElem[30]
                    collector_t2_fluct += outputElem[31]
                    collector_t2_rng += outputElem[32]
                    collector_u_star_1 += outputElem[33]
                    collector_Cd_coare_1 += outputElem[34]
                    collector_Cd_coare_2 += outputElem[35]
                    collector_zu1 += outputElem[36]
                    collector_zu2 += outputElem[37]
                    collector_laser1 += outputElem[38]
                    collector_laser2 += outputElem[39]
                    collector_laser3 += outputElem[40]
                    collector_laser4 += outputElem[41]

    write_message("Analysis run done!", filename='analysis_log.txt')
    return pd.DataFrame({"time": collector_time, "tauApprox_1": collector_tauApprox_1, "tauCoare_1": collector_tauCoare_1,
                            "Cd_1": collector_Cd_1, "U_anem_1": collector_U_anem_1, "HApprox_1": collector_HApprox_1, "HCoare_1": collector_HCoare_1, 
                            "u1": collector_u1, "u1_turb": collector_u1_turb, "v1": collector_v1, "v1_turb": collector_v1_turb, "w1": collector_w1, "w1_turb": collector_w1_turb,
                            "ta_1": collector_t1, "u2": collector_u2, "u2_turb": collector_u2_turb, "v2": collector_v2, "v2_turb": collector_v2_turb, "w2": collector_w2, "w2_turb": collector_w2_turb, 
                            "ta_2": collector_t2, "rho": collector_rho, "is_temp1_fluctuating": collector_t1_fluct, "is_temp1_range_large": collector_t1_rng, 
                            "is_temp2_fluctuating": collector_t2_fluct, "is_temp2_range_large": collector_t2_rng, "u_star_1": collector_u_star_1,
                            "tauApprox_2": collector_tauApprox_2, "tauCoare_2": collector_tauCoare_2, "Cd_2": collector_Cd_2, 
                            "U_anem_2": collector_U_anem_2, "HApprox_2": collector_HApprox_2, "HCoare_2": collector_HCoare_2, "u_star_2": collector_u_star_2,
                            "Cd_coare_1": collector_Cd_coare_1, "Cd_coare_2": collector_Cd_coare_2, "laser1": collector_laser1, "laser2": collector_laser2,
                            "laser3": collector_laser3, "laser4": collector_laser4, "zu_1": collector_zu1, "zu_2": collector_zu2})

def _analysis_iteration(file: Path, eraDf: pd.DataFrame, remsDf: pd.DataFrame, era_only=False, no_era=False) -> None:
    """
    Internal function which runs an iteration of an analysis run. Iterated externally by analysis_loop.
    """
    fileName = file.stem
    date = fileName[7:15]
    day = date[0:2]
    month = date[2:4]
    year = date[4:8]
    hour = fileName[16:18]
    
    # Defining constants
    TS_DEPTH = remsDf.depth[0] # Note that depth has to be extracted before we select the corresponding day as sometimes REMS may not exist on that day

    # Getting the corresponding day in the REMS data
    remsDf = remsDf.loc[(remsDf.timemet.map(lambda x: x.day) == int(day)) & (remsDf.timemet.map(lambda x: x.hour) == int(hour)) & (remsDf.timemet.map(lambda x: x.month) == int(month)) & (remsDf.timemet.map(lambda x: x.year) == int(year))]
    remsDf = remsDf.reset_index()
    eraDf = eraDf.loc[(eraDf.timemet.map(lambda x: x.day) == int(day)) & (eraDf.timemet.map(lambda x: x.hour) == int(hour)) & (eraDf.timemet.map(lambda x: x.month) == int(month)) & (eraDf.timemet.map(lambda x: x.year) == int(year))]
    eraDf = eraDf.reset_index()

    # Getting TIME_INTERVAL minute long slices and using them to get turbulent avg data over that same time frame
    data = DataAnalyser(file)

    slices = get_time_slices(data.df, TIME_INTERVAL)

    time_list = []
    rho_mean = []
    is_temp1_fluctuating = []
    is_temp1_range_large = []
    is_temp2_fluctuating = []
    is_temp2_range_large = []
    laser1_mean = []
    laser2_mean = []
    laser3_mean = []
    laser4_mean = []

    tau_approx_1 = []
    tau_coare_1 = []
    H_approx_1 = []
    H_coare_1 = []
    C_d_1 = []
    C_d_coare_1 = []
    U_anem_1_mean = []
    u_star_1_list = []
    u1_mean = []
    u1_turb_mean = []
    v1_mean = []
    v1_turb_mean = []
    w1_mean = []
    w1_turb_mean = []
    t1_mean = []
    zu1_mean = []

    tau_approx_2 = []
    tau_coare_2 = []
    H_approx_2 = []
    H_coare_2 = []
    C_d_2 = []
    C_d_coare_2 = []
    U_anem_2_mean = []
    u_star_2_list = []
    u2_mean = []
    u2_turb_mean = []
    v2_mean = []
    v2_turb_mean = []
    w2_mean = []
    w2_turb_mean = []
    t2_mean = []
    zu2_mean = []

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
    laser1 = 'Laser #1 Range (m)'
    laser2 = 'Laser #2 Range (m)'
    laser3 = 'Laser #3 Range (m)'
    laser4 = 'Laser #4 Range (m)'

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

    for slice in slices:
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
            
            eraSliceTemp = eraDf.loc[(time <= eraDf.timemet) & (eraDf.timemet <= time + datetime.timedelta(minutes=TIME_INTERVAL))]
            eraSliceTemp = eraSliceTemp.mean(numeric_only=True)
            if pd.notna(eraSliceTemp.loc['index']):
                eraSlice = eraSliceTemp # Guarding against ERA5's hour resolution from resulting in NaNs when incrementing up by less than 1hr at a time

        original_len = len(slice)
        slice = slice[~slice.is_temp1_range_large] # Removing erroneous points
        if len(slice)/original_len <= MIN_COV_SIZE:
            write_message(f'Too much cut out: {len(slice)}/{original_len}. {fileName} rejected.', filename='analysis_log.txt')
            continue

        # Flipping direction in anem 2 (TODO REMOVE AFTER NEXT FULLSWEEP)
        slice[u2] = -slice[u2]
        slice[v2] = -slice[v2]

        # Getting parameters
        jd = time - datetime.datetime(2015, 1, 1)
        jd = float(jd.days)
        tair = dataSlice.ta # NOTE COARE uses REMS temp (which is higher up)
        rh = dataSlice.rh
        p = dataSlice.press
        tsea = dataSlice.tsea
        sw_dn = dataSlice.solrad
        if not era_and_rems: lw_dn = dataSlice.thermrad # Only available with ERA5
        spechum = dataSlice.spech
        e = hum.hum2ea_modified(p, spechum)
        rho = hum.rhov_modified(tair, p, sh=spechum)

        # Calculating EC data (anem 1 is motion corrected)
        U_anem_1, U_anem_1_turb, w_vel_1, w_turb_1, T_turb_1 = get_windspeed_data(slice, u1, v1, w1, t1, mru_correct=True)
        U_anem_2, U_anem_2_turb, w_vel_2, w_turb_2, T_turb_2 = get_windspeed_data(slice, u2, v2, w2, t2, mru_correct=True)

        u_star_1 = np.sqrt(-get_covariance(U_anem_1_turb, w_turb_1))
        tau_approx_1.append(rho*(u_star_1**2))
        H_approx_1.append(rho*CPD*get_covariance(w_turb_1, T_turb_1))

        u_star_2 = np.sqrt(-get_covariance(U_anem_2_turb, w_turb_2))
        tau_approx_2.append(rho*(u_star_2**2))
        H_approx_2.append(rho*CPD*get_covariance(w_turb_2, T_turb_2))

        # Logging values
        u_star_1_list.append(u_star_1)
        U1_mean = np.mean(U_anem_1)
        C_d_1.append((u_star_1/U1_mean)**2)
        U_anem_1_mean.append(U1_mean)
        u1_mean.append(np.mean(slice[u1]))
        u1_turb_mean.append(np.mean(get_turbulent(slice[u1])))
        v1_mean.append(np.mean(slice[v1]))
        v1_turb_mean.append(np.mean(get_turbulent(slice[v1])))
        w1_mean.append(np.mean(w_vel_1))
        w1_turb_mean.append(np.mean(w_turb_1))
        t1_mean.append(np.mean(slice[t1]))

        u_star_2_list.append(u_star_2)
        U2_mean = np.mean(U_anem_2)
        C_d_2.append((u_star_2/U2_mean)**2)
        U_anem_2_mean.append(U2_mean)
        u2_mean.append(np.mean(slice[u2]))
        u2_turb_mean.append(np.mean(get_turbulent(slice[u2])))
        v2_mean.append(np.mean(slice[v2]))
        v2_turb_mean.append(np.mean(get_turbulent(slice[v2])))
        w2_mean.append(np.mean(w_vel_2))
        w2_turb_mean.append(np.mean(w_turb_2))
        t2_mean.append(np.mean(slice[t2]))

        rho_mean.append(np.mean(rho))
        is_temp1_fluctuating.append(slice.is_temp1_fluctuating.any())
        is_temp1_range_large.append(slice.is_temp1_range_large.any())
        is_temp2_fluctuating.append(slice.is_temp2_fluctuating.any())
        is_temp2_range_large.append(slice.is_temp2_range_large.any())

        if slice[laser1] is None:
            l1 = l2 = l3 = l4 = ZT
        else:
            l1 = np.mean(slice[laser1])
            l2 = np.mean(slice[laser2])
            l3 = np.mean(slice[laser3])
            l4 = np.mean(slice[laser4])

        laser1_mean.append(l1)
        laser2_mean.append(l2)
        laser3_mean.append(l3)
        laser4_mean.append(l4)

        # Getting COARE's predictions
        zu_1 = l1 - LASER_TO_ANEM_1
        zu1_mean.append(zu_1)
        coare_res = get_coare_data(U1_mean, jd, zu_1, tair, rh, p, tsea, sw_dn, TS_DEPTH)
        if coare_res is None:
            write_message(f"ANEM 1 ERROR IN {fileName}: SKIPPED FOR NOW", filename='analysis_log.txt')
            tau_coare_1.append(np.nan)
            H_coare_1.append(np.nan)
            C_d_coare_1.append(np.nan)
        else:
            tau_coare_1.append(coare_res[1])
            H_coare_1.append(coare_res[2])
            C_d_coare_1.append(coare_res[12])

        zu_2 = l1 - LASER_TO_ANEM_2
        zu2_mean.append(zu_2)
        coare_res = get_coare_data(U2_mean, jd, zu_2, tair, rh, p, tsea, sw_dn, TS_DEPTH)
        if coare_res is None:
            write_message(f"ANEM 2 ERROR IN {fileName}: SKIPPED FOR NOW", filename='analysis_log.txt')
            tau_coare_2.append(np.nan)
            H_coare_2.append(np.nan)
            C_d_coare_2.append(np.nan)
        else:
            tau_coare_2.append(coare_res[1])
            H_coare_2.append(coare_res[2])
            C_d_coare_2.append(coare_res[12])

        # Updating time
        time_list.append(time)
        time += datetime.timedelta(minutes=TIME_INTERVAL)
    
    if era_and_rems:
        write_message(f"Analysed {fileName} with REMS", filename='analysis_log.txt')
    else:
        write_message(f"Analysed {fileName} with ERA5", filename='analysis_log.txt')

    return (tau_approx_1, tau_coare_1, C_d_1, U_anem_1_mean, H_approx_1, H_coare_1, time_list, 
            u1_mean, u1_turb_mean, v1_mean, v1_turb_mean, w1_mean, w1_turb_mean, t1_mean,
            tau_approx_2, tau_coare_2, H_approx_2, H_coare_2, C_d_2, U_anem_2_mean, u_star_2_list,
            u2_mean, u2_turb_mean, v2_mean, v2_turb_mean, w2_mean, w2_turb_mean, 
            t2_mean, rho_mean, is_temp1_fluctuating, is_temp1_range_large,
            is_temp2_fluctuating, is_temp2_range_large, u_star_1_list, C_d_coare_1, C_d_coare_2, zu1_mean,
            zu2_mean, laser1_mean, laser2_mean, laser3_mean, laser4_mean)

def get_coare_data(U_mean: float, jd: float, zu: float, tair: float, rh: float, p: float, tsea: float, sw_dn: float, 
                   ts_depth: float) -> np.ndarray:
    # TODO: zrf_u, etc. NEEDS TO BE SET TO ANEM HEIGHT INITIALLY, THEN WE CAN LIN INTERP TO 10m
    try:
        blockPrint()
        coare_res = coare(Jd=jd, U=U_mean, Zu=zu, Tair=tair, Zt=ZT, RH=rh, Zq=ZQ, P=p, 
                            Tsea=tsea, SW_dn=sw_dn, LW_dn=LW_DN, Lat=LAT, Lon=LON, Zi=ZI, 
                            Rainrate=RAINRATE, Ts_depth=ts_depth, Ss=SS, cp=None, sigH=None,
                            zrf_u=zu, zrf_t=zu, zrf_q=zu)
        enablePrint()
    except Exception as e:
        return None

    return coare_res[0]

def get_H_err_coeffs(slice: pd.DataFrame, u: str, t: str, tair: float) -> np.ndarray:
    # Burns et. al
    u_anem = slice[u]
    temp_diff = tair - slice[t]
    try:
        p = np.polyfit(u_anem, temp_diff, deg=3)
    except np.linalg.LinAlgError:
        return np.nan*np.ones(len(u_anem))
    return np.zeros(len(u_anem)) #3*p[0]*u_anem**2 + 2*p[1]*u_anem + p[2]

def get_windspeed_data(slice: pd.DataFrame, u: str, v: str, w: str, t: str, mru_correct=True) -> tuple:
    # Getting current-corrected windspeed
    U_mag = np.sqrt(slice[u]**2 + slice[v]**2)
    w_vel = slice[w]
    # Easterly -> +ive x axis, Northerly -> +ive y. Note that anem v+ is west so east is -v
    # u_AirWat = u_Air - u_Wat
    #U_vec.East = U_vec.East - remsSlice.cur_e_comp # Seem to be negligible compared to wind speed
    #U_vec.North = U_vec.North - remsSlice.cur_n_comp
    # u = np.sqrt(U_10_vec.North**2 + U_10_vec.East**2) #TODO CHANGE TO U_10_mag
    # U_vec = pd.DataFrame({'East': slice[v], 'North': slice[u]}) # NOTE LOCALLY MRU UNCORRECTED
    # U_vec = U_vec.mean() #Taking TIME_INTERVAL min avg

    # Locally MRU correcting
    if mru_correct:
        theta = np.arctan2(np.mean(w_vel), np.mean(U_mag))
        U_mag_corr = w_vel*np.sin(theta) + U_mag*np.cos(theta)
        w_vel_corr = w_vel*np.cos(theta) - U_mag*np.sin(theta)

        U_mag = U_mag_corr
        w_vel = w_vel_corr

    # Getting magnitude of turbulent horizontal velocity vector
    U_turb = get_turbulent(U_mag)
    
    w_turb = get_turbulent(w_vel)
    T_turb = get_turbulent(slice[t])
    # T_turb = get_turbulent(slice[t]/(1 + 0.378*e/p))

    return U_mag, U_turb, w_vel, w_turb, T_turb

def get_time_slices(df: pd.DataFrame, interval_min: float) -> list:
    """
    Breaks df up into interval_min minute long intervals which are compiled in a list
    """
    window_width = round((interval_min*60)/(df.GlobalSecs[1] - df.GlobalSecs[0])) # Amount of indicies to consider = wanted_stepsize/data_stepsize
    slices = df.rolling(window=window_width, step=window_width)

    return [slice for slice in slices if len(slice) >= MIN_SLICE_SIZE]

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
    if len(u) <= 1 or len(v) <= 1:
        write_message(f'Covariance cannot be calculated on an input shorter than length 2.', filename='analysis_log.txt')
        return np.nan

    logical = ((pd.notna(u)) & (pd.notna(v)))

    if len(logical)/len(u) <= MIN_COV_SIZE:
        write_message(f'Timeseries too short for reasonable covariance calc: {round(100*len(logical)/len(u),2)}%', filename='analysis_log.txt')
        return np.nan
    
    u = u[logical]
    v = v[logical]

    if len(u) != len(v):
        write_message(f'Both inputs must be the same length for covariance.', filename='analysis_log.txt')
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

        # Very rudimentary "derivative"
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

    sns.lineplot(data=outDf, x='time', y='laser1', label='laser 1')
    sns.lineplot(data=outDf, x='time', y='laser2', label='laser 2')
    sns.lineplot(data=outDf, x='time', y='laser3', label='laser 3')
    sns.lineplot(data=outDf, x='time', y='laser4', label='laser 4')
    plt.xlabel('time')
    plt.ylabel('Sea Level from Lasers (m) (28m AMSL)')
    plt.ylim([20, 30])
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'laser.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='zu_1', label='Anem 1')
    sns.lineplot(data=outDf, x='time', y='zu_2', label='Anem 2')
    plt.xlabel('time')
    plt.ylabel('Anemometer Height Above Sea Level (m)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'zu.png'))
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

    sns.lineplot(data=outDf, x='time', y='U_anem_1', label='Anem 1')
    sns.lineplot(data=outDf, x='time', y='U_anem_2', label='Anem 2')
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

    lin_lims = [min([min(outDf.tauCoare_1), min(outDf.tauApprox_1)]), max([max(outDf.tauCoare_1), max(outDf.tauApprox_1)])]
    sns.regplot(data=outDf, x='tauCoare_1', y='tauApprox_1', label='Anem 1 Best fit with 95% CI')
    sns.regplot(data=outDf, x='tauCoare_2', y='tauApprox_2', label='Anem 2 Best fit with 95% CI')
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Shear Stress')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_xy.png'))
        plt.close()
    else:
        plt.show()   

    lin_lims = [min([min(outDf.tauCoare_1), min(outDf.tauApprox_1)]), max([max(outDf.tauCoare_1), max(outDf.tauApprox_1)])]
    # sns.kdeplot(data=outDf, x='tauCoare_1', y='tauApprox_1', fill=True, levels=100, cmap='mako', thresh=0)
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Anem 1 Shear Stress')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_xy.png'))
        plt.close()
    else:
        plt.show() 

    lin_lims = [min([min(outDf.tauCoare_2), min(outDf.tauApprox_2)]), max([max(outDf.tauCoare_2), max(outDf.tauApprox_2)])]
    # sns.kdeplot(data=outDf, x='tauCoare_2', y='tauApprox_2', fill=True, levels=100, cmap='mako', thresh=0)
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Anem 2 Shear Stress')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_xy.png'))
        plt.close()
    else:
        plt.show() 

    lin_lims = [min([min(outDf.HCoare_1), min(outDf.HApprox_1)]), max([max(outDf.HCoare_1), max(outDf.HApprox_1)])]
    sns.regplot(data=outDf, x='HCoare_1', y='HApprox_1', label='Anem 1 Best fit with 95% CI')
    sns.regplot(data=outDf, x='HCoare_2', y='HApprox_2', label='Anem 2 Best fit with 95% CI')
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Sensible Heat Flux (Wm^-2)')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_xy.png'))
        plt.close()
    else:
        plt.show()   

    lin_lims = [min([min(outDf.HCoare_1), min(outDf.HApprox_1)]), max([max(outDf.HCoare_1), max(outDf.HApprox_1)])]
    # sns.kdeplot(data=outDf, x='HCoare_1', y='HApprox_1', fill=True, levels=100, cmap='mako', thresh=0)
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Anem 1 Sensible Heat Flux (Wm^-2)')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_xy.png'))
        plt.close()
    else:
        plt.show()  

    lin_lims = [min([min(outDf.HCoare_2), min(outDf.HApprox_2)]), max([max(outDf.HCoare_2), max(outDf.HApprox_2)])]
    # sns.kdeplot(data=outDf, x='HCoare_2', y='HApprox_2', fill=True, levels=100, cmap='mako', thresh=0)
    sns.lineplot(x=lin_lims, y=lin_lims, label='1:1 Fit')
    plt.xlabel('COARE')
    plt.ylabel('EC')
    plt.title('Anem 2 Sensible Heat Flux (Wm^-2)')
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_xy.png'))
        plt.close()
    else:
        plt.show()   

    sns.lineplot(data=outDf, x='time', y='u_star_1', label="Anem 1 u_star", marker='.')
    sns.lineplot(data=outDf, x='time', y='u_star_2', label="Anem 2 u_star", marker='.')
    plt.xlabel('time')
    plt.ylabel('u_star (m/s)')
    plt.xticks(plt.xticks()[0], rotation=90)
    plt.show()

    # Making a box for x [0, 25], y [-2, 5]
    # left_wall = [[0, 0], [-2, 5]]
    # right_wall = [[25, 25], [-2, 5]]
    # bottom_wall = [[0, 25], [-2, -2]]
    # top_wall = [[0, 25], [5, 5]]
    # sns.lineplot(x=left_wall[0], y=left_wall[1], color='black')
    # sns.lineplot(x=right_wall[0], y=right_wall[1], color='black')
    # sns.lineplot(x=bottom_wall[0], y=bottom_wall[1], color='black')
    # sns.lineplot(x=top_wall[0], y=top_wall[1], color='black')
    sns.scatterplot(x=outDf.U_anem_1, y=1000*outDf.Cd_1, label='Anem 1')
    sns.lineplot(x=outDf.U_anem_1, y=1000*outDf.Cd_coare_1, label='COARE', color='black')
    plt.xlabel('U_10 (Approx) (m/s)')
    plt.ylabel('Cd*10^3')
    plt.ylim([-2,5]) # Limits as per box dimensions
    plt.xlim([0, 25])
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'Cd_spread.png'))
        plt.close()
    else:
        plt.show()   

    sns.scatterplot(x=outDf.U_anem_2, y=1000*outDf.Cd_2, label='Anem 2')
    sns.lineplot(x=outDf.U_anem_2, y=1000*outDf.Cd_coare_2, label='COARE', color='black')
    plt.xlabel('U_10 (Approx) (m/s)')
    plt.ylabel('Cd*10^3')
    plt.xlim([0, 25])
    plt.ylim([-2,5]) # Limits as per box dimensions
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'Cd_spread.png'))
        plt.close()
    else:
        plt.show()  

    mean_ec = apply_window_wise(outDf.tauApprox_1, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='tauApprox_1', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='tauCoare_1', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Anem 1 Shear Stress (Nm^-2)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_timeseries.png'))
        plt.close()
    else:
        plt.show()  

    mean_ec = apply_window_wise(outDf.tauApprox_2, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='tauApprox_2', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='tauCoare_2', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Anem 2 Shear Stress (Nm^-2)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'tau_timeseries.png'))
        plt.close()
    else:
        plt.show()

    mean_ec = apply_window_wise(outDf.HApprox_1, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='HApprox_1', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='HCoare_1', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Anem 1 Sensible Heat Flux (Wm^-2)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show()  

    mean_ec = apply_window_wise(outDf.HApprox_2, WINDOW_WIDTH, np.mean)
    sns.scatterplot(data=outDf, x='time', y='HApprox_2', marker='.', color='blue', label='EC')
    sns.lineplot(data=outDf, x='time', y='HCoare_2', color='orange', label='COARE')
    sns.scatterplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green', label='Mean EC')
    sns.lineplot(x=outDf.time[::WINDOW_WIDTH], y=mean_ec, color='green')
    plt.xlabel('time')
    plt.ylabel('Anem 2 Sensible Heat Flux (Wm^-2)')
    plt.xticks(plt.xticks()[0], rotation=90)
    if save_plots:
        plt.savefig(os.path.join(writeDir, 'Postprocess', 'H_timeseries.png'))
        plt.close()
    else:
        plt.show()    

def aggregate_dfs(dir: Path, keyword: str):
    df = pd.DataFrame()
    for file in dir.iterdir():
        if keyword in file.stem and 'csv' in file.suffix:
            df_to_cat = pd.read_csv(file)
            pd.concat([df, df_to_cat])
            os.remove(file)

    try:    
        df.sort_values(by='time')
    except KeyError:
        df.sort_values(by='timemet')

    df.to_csv(dir / f'{keyword}_aggregated.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', nargs='+', type=str, help='Path to the rawdata. Can be a list.')
    parser.add_argument('--write_dir', type=str, help='Path to output.')
    parser.add_argument('--cpu_fraction', type=float, help='% Of CPUs to use. Can be within (0,1].', default=1)
    parser.add_argument('--run_supervised', action='store_true', help='Run one-by-one analysis', default=False)
    parser.add_argument('--era_only', action='store_true', help='If True, always use ERA5 and never use REMS for relevant parameters. If False, will use REMS when available and ERA5 otherwise.', default=False)
    parser.add_argument('--no_era', action='store_true', help='If True, will never use ERA5 - only REMS (skips unavailable times).', default=False)
    parser.add_argument('--era_filename', type=str, help='Name of ERA5 npz file (e.g. era2015.npz)')
    args = parser.parse_args()

    # Using a modified version of np.load to read data with allow_pickle turned off in the .npz file
    np_load_modified = lambda *a,**k: np.load(*a, allow_pickle=True, **k)

    # Grabbing REMS stuff
    for cyclone in ['2015']:
        with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'REMS', f'meteo_{cyclone}.npz')) as metFile:
            timemet = metFile['timemet.npy'] # YYYYMMDD and milliseconds past midnight
            press = metFile['press.npy'] # Barometric Pressure (hPa=mbar)
            rh = metFile['rh.npy'] # Relative Humidity (%)
            spech = metFile['spech.npy'] # Specific humidity (rh: ratio, p: Pa; T: Kelvin)
            ta = metFile['ta.npy'] # Air Temperature (C)
            solrad = metFile['solrad.npy'] # Downward Solar radiation (Wm^-2)
        with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'REMS', f'meteo_{cyclone}_currents.npz')) as metFile:
            timemet_currents = metFile['timemet.npy']
            cur_n_comp = metFile['cur_n_comp.npy'] # Northward component of current velocity (m/s)
            cur_e_comp = metFile['cur_e_comp.npy'] # Eastward component of current velocity (m/s)
            tsea = metFile['tsea.npy'] # Water temperature (degC)
            depth = metFile['depth.npy'] # Approx. distance from surface (m), Babanin et al.

    # The current and meteo arrays may not be of the same length
    master_len = len(timemet) + len(timemet_currents)
    master_arr = np.zeros((master_len, 10))
    master_arr[:len(timemet)] = np.concatenate((timemet, press, rh, spech, ta, solrad), axis=1)
    i = j = 0
    while i < len(master_arr) and j < len(timemet_currents):
        if master_arr[i, 0] == timemet_currents[j] or master_arr[i, 0] == 0:
            master_arr[i, :] = np.array((cur_n_comp[j], cur_e_comp[j], tsea[j], depth[j]))
            j += 1
        i += 1
    master_arr = master_arr[master_arr[:, 0] != 0]

    remsDf = pd.DataFrame({"timemet": master_arr[:, 0], "press": master_arr[:, 1], "rh": master_arr[:, 2], 
                        "spech": master_arr[:, 3], "ta": master_arr[:, 4], "solrad": master_arr[:, 5], 
                        "cur_n_comp": master_arr[:, 6], "cur_e_comp": master_arr[:, 7], 
                        "tsea": master_arr[:, 8], "depth": master_arr[:, 9]})
    # remsDf = pd.DataFrame({"timemet": timemet, "press": press, "rh": rh, "spech": spech, "ta": ta, "solrad": solrad,
    #                         "cur_n_comp": cur_n_comp, "cur_e_comp": cur_e_comp, "tsea": tsea, "depth": depth})

    # Grabbing ERA5 data
    with np_load_modified(os.path.join(os.getcwd(), 'Resources', 'ERA5', args.era_filename)) as eraFile:
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
    writeDir = Path(args.write_dir)
    for i, _ in enumerate(args.read_dir):
        readDir = Path(args.read_dir[i])

        # Making folders
        os.mkdir(os.path.join(writeDir, 'Preprocess'))
        os.mkdir(os.path.join(writeDir, 'Preprocess', 'REMS vs ERA'))
        os.mkdir(os.path.join(writeDir, 'Postprocess'))

        eraDf, remsDf = preprocess(eraDf, remsDf, writeDir=writeDir, era_only=args.era_only)
        outDf = analysis_loop(readDir, eraDf, remsDf, supervised=args.run_supervised, cpuFraction=args.cpu_fraction, era_only=args.era_only, no_era=args.no_era)

        outDf = outDf.sort_values(by='time') # Sorting outDf since it may be jumbled due to multiprocessing

        eraDf.to_csv(os.path.join(writeDir, f'eraDf_{readDir.stem}.csv'))
        remsDf.to_csv(os.path.join(writeDir, f'remsDf_{readDir.stem}.csv'))
        outDf.to_csv(os.path.join(writeDir, f'outDf_{readDir.stem}.csv'))

    # If multiple months have been analysed, combine them into a single csv
    if len(args.read_dir) > 1:
        aggregate_dfs(writeDir, keyword='eraDf')
        aggregate_dfs(writeDir, keyword='remsDf')
        aggregate_dfs(writeDir, keyword='outDf')

    t1 = time.perf_counter()
    
    write_message(f"Took {round((t1-t0)/60, 1)}min", filename='analysis_log.txt')