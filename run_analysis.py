import os
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import MeterologicalParameters.humidity as hum
import matplotlib.pyplot as plt
import glob
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

# Default parameters
LW_DN = 370
ZI = 600
RAINRATE = 0

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def analysis_loop(readDir: Path, remsDf: pd.DataFrame, eraDf: pd.DataFrame, supervised=True, cpuFraction=1, era_only=False, no_era=False) -> pd.DataFrame:
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
    collector_U10 = []
    collector_w_turb = []
    collector_u = []
    collector_v = []
    collector_w = []
    collector_t = []
    collector_rho = []

    # One-by-one
    if supervised:
        for file in files:
            output = _analysis_iteration(file, remsDf, eraDf, era_only, no_era)
            if output is not None:
                collector_tauApprox += output[0]
                collector_tauCoare += output[1]
                collector_Cd += output[2]
                collector_U10 += output[3]
                collector_HApprox += output[4]
                collector_HCoare += output[5]
                collector_w_turb += output[6]
                collector_time += output[7]
                collector_u += output[8]
                collector_v += output[9]
                collector_w += output[10]
                collector_t += output[11]
                collector_rho += output[12]

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
        args = [*zip(files, remsArr, eraArr, eraOnlyArr, noEraArr)]

        with mp.Pool(coresToUse) as p:
            output = p.starmap(_analysis_iteration, iterable=args)
            for outputElem in output:
                if outputElem is not None:
                    collector_tauApprox += outputElem[0]
                    collector_tauCoare += outputElem[1]
                    collector_Cd += outputElem[2]
                    collector_U10 += outputElem[3]
                    collector_HApprox += outputElem[4]
                    collector_HCoare += outputElem[5]
                    collector_w_turb += outputElem[6]
                    collector_time += outputElem[7]
                    collector_u += outputElem[8]
                    collector_v += outputElem[9]
                    collector_w += outputElem[10]
                    collector_t += outputElem[11]
                    collector_rho += outputElem[12]

    write_message("Analysis run done!", filename='analysis_log.txt')
    return pd.DataFrame({"time": collector_time, "tauApprox": collector_tauApprox, "tauCoare": collector_tauCoare,
                            "Cd": collector_Cd, "U10": collector_U10, "HApprox": collector_HApprox, "HCoare": collector_HCoare, 
                            "wTurb": collector_w_turb, "u": collector_u, "v": collector_v, "w": collector_w, "ta": collector_t,
                            "rho": collector_rho})

def _analysis_iteration(file: Path, remsDf: pd.DataFrame, eraDf: pd.DataFrame, era_only=False, no_era=False) -> None:
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

    # Getting time_interval minute long slices and using them to get turbulent avg data over that same time frame
    time_interval = 10
    data = DataAnalyser(file)
    slices = get_time_slices(data.df, time_interval)

    # NOTE: FILL IN AS REQUIRED
    time_list = pd.Series(np.zeros(len(slices)))
    tau_approx = pd.Series(np.zeros(len(slices)))
    tau_coare = pd.Series(np.zeros(len(slices)))
    H_approx = pd.Series(np.zeros(len(slices)))
    H_coare = pd.Series(np.zeros(len(slices)))
    C_d = pd.Series(np.zeros(len(slices)))
    U_10_mag = pd.Series(np.zeros(len(slices))) # NOTE: "_mag" is to prevent it being const from all caps
    w_turb_list = pd.Series(np.zeros(len(slices)))
    u_mean = pd.Series(np.zeros(len(slices)))
    v_mean = pd.Series(np.zeros(len(slices)))
    w_mean = pd.Series(np.zeros(len(slices)))
    t_mean = pd.Series(np.zeros(len(slices)))
    rho_mean = pd.Series(np.zeros(len(slices)))

    w2 = "Anemometer #1 W Velocity (ms-1)"
    u2 = "Anemometer #1 U Velocity (ms-1)"
    v2 = "Anemometer #1 V Velocity (ms-1)"
    t2 = "Anemometer #1 Temperature (degC)"
    comp2 = "Compass #1 (deg)"

    # Using ERA5
    if (era_only or len(remsDf) == 0) and not no_era:
        time = eraDf.timemet[0]
        for i, slice in enumerate(slices):
            # Getting the REMS data for the particular time interval and using the average
            eraSliceTemp = eraDf.loc[(time <= eraDf.timemet) & (eraDf.timemet <= time + datetime.timedelta(minutes=time_interval))]
            eraSliceTemp = eraSliceTemp.mean(numeric_only=True)
            if pd.notna(eraSliceTemp.loc['index']):
                eraSlice = eraSliceTemp # Guarding against ERA5's hour resolution from resulting in NaNs when incrementing up by less than 1hr at a time
            else:
                pass

            # TODO: Correcting for POSSIBLE error in anem temp (10degC hotter than REMS)
            #slice[t2] = slice[t2] - 5
            #slice[u2] = -slice[u2]
            #slice[v2] = -slice[v2]

            # Getting constants
            jd = time - datetime.datetime(2015, 1, 1)
            jd = float(jd.days)
            tair = eraSlice.ta
            rh = eraSlice.rh
            p = eraSlice.press
            tsea = eraSlice.tsea
            sw_dn = eraSlice.solrad
            lw_dn = eraSlice.thermrad
            spechum = eraSlice.spech
            #e = hum.hum2ea_modified(p, spechum)
            rho = hum.rhov_modified(tair, p, sh=spechum)

            w2_turb = get_turbulent(slice[w2])
            T2_turb = get_turbulent(slice[t2])
            #T2_turb = T2_turb/(1 + 0.378*e/p)
            w_turb_list[i] = np.mean(w2_turb*T2_turb)

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

            #u_star_2 = np.mean(-U2_turb*w2_turb)
            u_star_2 = np.mean(U2_turb*w2_turb) - np.mean(U2_turb)*np.mean(w2_turb)
            
            tau_approx[i] = -rho*u_star_2
            #H_approx[i] = rho*hum.cpd*np.mean(w2_turb*T2_turb)
            H_approx[i] = rho*hum.cpd*(np.mean(w2_turb*T2_turb) - np.mean(w2_turb)*np.mean(T2_turb))

            #TODO: Assume U_10 ~= U_14.8 for now
            C_d[i] = np.mean(-U2_turb*w2_turb)/(np.mean(U2_mag)**2)
            #C_d[i] = -np.cov([U2_turb.mean(), w2_turb.mean()])/np.mean(U2_mag)
            U_10_mag[i] = np.mean(U2_mag)
            u_mean[i] = np.mean(slice[u2])
            v_mean[i] = np.mean(slice[v2])
            w_mean[i] = np.mean(slice[w2])
            t_mean[i] = np.mean(slice[t2])
            rho_mean[i] = np.mean(rho)
            

            # TODO: zrf_u, etc. NEEDS TO BE SET TO ANEM HEIGHT INITIALLY, THEN WE CAN LIN INTERP TO 10m
            try:
                blockPrint()
                coare_res = coare(Jd=jd, U=u, Zu=ZU, Tair=tair, Zt=ZT, RH=rh, Zq=ZQ, P=p, Tsea=tsea, SW_dn=sw_dn, LW_dn=LW_DN, Lat=LAT, Lon=LON, Zi=ZI, Rainrate=RAINRATE, Ts_depth=TS_DEPTH, Ss=SS, cp=None, sigH=None,zrf_u = ZU,zrf_t = ZU,zrf_q = ZU)
                enablePrint()
                tau_coare[i] = coare_res[0][1]
                H_coare[i] = coare_res[0][2]
            except IndexError:
                write_message(f"ERROR IN {fileName} - SKIPPED FOR NOW", filename='analysis_log.txt')

            # Updating time
            time_list[i] = time
            time += datetime.timedelta(minutes=time_interval)

            # Investigating the streak
            if tau_approx[i]/tau_coare[i] >= 2/0.5 and tau_approx[i] >= 1.5:
               write_message(f"tau spike in {fileName}", filename='analysis_log.txt')
        
        write_message(f"Analysed {fileName} with ERA5", filename='analysis_log.txt')

        return (tau_approx.to_list(), tau_coare.to_list(), C_d.to_list(), U_10_mag.to_list(), H_approx.to_list(), H_coare.to_list(), 
                w_turb_list.to_list(), time_list.to_list(), u_mean.to_list(), v_mean.to_list(), w_mean.to_list(), t_mean.to_list(),
                rho_mean.to_list())

    # Using REMS
    elif len(remsDf) != 0:
        time = remsDf.timemet[0]
        for i, slice in enumerate(slices):
            # Getting the REMS data for the particular time interval and using the average
            remsSlice = remsDf.loc[(time <= remsDf.timemet) & (remsDf.timemet <= time + datetime.timedelta(minutes=time_interval))]
            remsSlice = remsSlice.mean(numeric_only=True)

            # TODO: Correcting for POSSIBLE error in anem temp (10degC hotter than REMS)
            # TODO NOT DONE FOR ANEM 2 IN CLEANUP
            #slice[t2] = slice[t2] - 5
            #slice[u2] = -slice[u2]
            #slice[v2] = -slice[v2]

            # Getting constants
            jd = time - datetime.datetime(2015, 1, 1)
            jd = float(jd.days)
            tair = remsSlice.ta
            rh = remsSlice.rh
            p = remsSlice.press # TODO need to correct for height
            tsea = remsSlice.tsea
            sw_dn = remsSlice.solrad
            spechum = remsSlice.spech
            #e = hum.hum2ea_modified(p, spechum)
            rho = hum.rhov_modified(tair, p, sh=spechum)

            w2_turb = get_turbulent(slice[w2])
            T2_turb = get_turbulent(slice[t2])
            #T2_turb = T2_turb/(1 + 0.378*e/p) 
            w_turb_list[i] = np.mean(w2_turb*T2_turb)

            # Getting magnitude of turbulent horizontal velocity vector
            U2_turb = get_turbulent(np.sqrt(slice[u2]**2 + slice[v2]**2))

            # Getting current-corrected windspeed
            U2_mag = np.sqrt(slice[u2]**2 + slice[v2]**2)
            # Easterly -> +ive x axis, Northerly -> +ive y.
            U2_vec = pd.DataFrame({'East': slice[v2], 'North': slice[u2]})
            U2_vec = U2_vec.mean() # Taking 10min avg

            # u_AirWat = u_Air - u_Wat
            U_vec = U2_vec
            #U_vec.East = U_vec.East - remsSlice.cur_e_comp
            #U_vec.North = U_vec.North - remsSlice.cur_n_comp
            u = np.sqrt(U_vec.North**2 + U_vec.East**2)

            #u_star_2 = np.mean(-U2_turb*w2_turb)
            u_star_2 = np.mean(U2_turb*w2_turb) - np.mean(U2_turb)*np.mean(w2_turb)
            
            tau_approx[i] = -rho*u_star_2
            #H_approx[i] = rho*hum.cpd*np.mean(w2_turb*T2_turb)
            H_approx[i] = rho*hum.cpd*(np.mean(w2_turb*T2_turb) - np.mean(w2_turb)*np.mean(T2_turb))

            # TODO: Assume U_10 ~= U_14.8 for now
            C_d[i] = np.mean(-U2_turb*w2_turb)/(np.mean(U2_mag)**2)
            #C_d[i] = -np.cov([U2_turb.mean(), w2_turb.mean()])/np.mean(U2_mag)
            U_10_mag[i] = np.mean(U2_mag)
            u_mean[i] = np.mean(slice[u2])
            v_mean[i] = np.mean(slice[v2])
            w_mean[i] = np.mean(slice[w2])
            t_mean[i] = np.mean(slice[t2])

            # TODO: zrf_u, etc. NEEDS TO BE SET TO ANEM HEIGHT INITIALLY, THEN WE CAN LIN INTERP TO 10m
            try:
                blockPrint()
                coare_res = coare.coare36vnWarm_et(Jd=jd, U=u, Zu=ZU, Tair=tair, Zt=ZT, RH=rh, Zq=ZQ, P=p, Tsea=tsea, SW_dn=sw_dn, LW_dn=LW_DN, Lat=LAT, Lon=LON, Zi=ZI, Rainrate=RAINRATE, Ts_depth=TS_DEPTH, Ss=SS, cp=None, sigH=None,zrf_u = ZU,zrf_t = ZU,zrf_q = ZU)
                enablePrint()
                tau_coare[i] = coare_res[0][1]
                H_coare[i] = coare_res[0][2]
            except:
                write_message(f"ERROR IN {fileName} - SKIPPED FOR NOW", filename='analysis_log.txt')

            # Updating time
            time_list[i] = time
            time += datetime.timedelta(minutes=time_interval)
    
            # Investigating the streak
            #if tau_approx[i]/tau_coare[i] >= 1/0.25 and tau_approx[i] >= 1.25:
            #    write_message(f"tau spike in {fileName}", filename='analysis_log.txt')
            #if H_approx[i] > 251 and H_approx[i] < 252 and H_coare[i] > 76 and H_coare[i] < 77:
            #    write_message(f"H spike in {fileName}", filename='analysis_log.txt')

        write_message(f"Analysed {fileName} with REMS", filename='analysis_log.txt')

        return (tau_approx.to_list(), tau_coare.to_list(), C_d.to_list(), U_10_mag.to_list(), H_approx.to_list(), H_coare.to_list(), 
                w_turb_list.to_list(), time_list.to_list(), u_mean.to_list(), v_mean.to_list(), w_mean.to_list(), t_mean.to_list(),
                rho_mean.to_list())

    # If there's no match with REMS and ERA5 isn't being used
    elif no_era:
        write_message(f"No date matches between {fileName} and REMS. ERA5 turned off.", filename='analysis_log.txt')
        return None

    else:
        raise ValueError("None of the analyses cases were triggered")     

def get_time_slices(df: pd.DataFrame, interval_min: float) -> list[pd.DataFrame]:
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

def postprocess() -> None:
    plt.plot(outDf.time, outDf.rho, "-o")
    plt.xlabel('time')
    plt.ylabel('Air Density (kg/m^3)')
    plt.show()
    
    plt.plot(outDf.time, outDf.HCoare, "-o")
    #plt.plot(outDf2.time, outDf2.HCoare, "-o")
    #plt.legend(['REMS', 'ERA5'])
    plt.xlabel('time')
    plt.ylabel('Sensible Heat Flux')
    plt.show()

    plt.plot(outDf.time, outDf.u)
    plt.plot(eraDf.timemet, eraDf.v_10)
    plt.legend(['Our data: u component (14.8m)', 'ERA5: v component (10m)'])
    plt.xlabel('time')
    plt.ylabel('Northerly component of wind speed (m/s)')
    plt.show()

    plt.plot(outDf.time, outDf.v)
    plt.plot(eraDf.timemet, eraDf.u_10)
    plt.legend(['Our data: v component (14.8m)', 'ERA5: u component (10m)'])
    plt.xlabel('time')
    plt.ylabel('Easterly component of wind speed (m/s)')
    plt.show()

    plt.plot(outDf.time, outDf.ta)
    plt.plot(eraDf.timemet, eraDf.ta)
    plt.xlabel('time')
    plt.ylabel('Sea Surface Temperature (degC)')
    plt.legend(['Anem 2', 'ERA5'])
    plt.show()

    plt.plot([min([min(outDf.tauCoare), min(outDf.tauApprox)]), max([max(outDf.tauCoare), max(outDf.tauApprox)])], [min([min(outDf.tauCoare), min(outDf.tauApprox)]), max([max(outDf.tauCoare), max(outDf.tauApprox)])], color="r")
    plt.scatter(outDf.tauCoare, outDf.tauApprox, c=outDf.U10)
    plt.colorbar(label='U_10 (m/s)')
    #plt.scatter(outDf.tauCoare, outDf.tauApprox, c=collector_w_turb)
    #plt.colorbar(label="w' (m/s)")
    plt.xlabel("COARE")
    plt.ylabel("rho*u_star^2")
    plt.title("Shear Stress")
    plt.show()
    
    for i in range(len(outDf.Cd)):
        outDf.Cd[i] = 1000*outDf.Cd[i]

    plt.scatter(outDf.U10, outDf.Cd)
    plt.xlim([0, 25])
    plt.ylim([-2,5])
    plt.xlabel('U_10 (m/s)')
    plt.ylabel('C_d*1000')
    plt.show()

    plt.plot([min([min(outDf.HCoare), min(outDf.HApprox)]), max([max(outDf.HCoare), max(outDf.HApprox)])], [min([min(outDf.HCoare), min(outDf.HApprox)]), max([max(outDf.HCoare), max(outDf.HApprox)])], color="r")
    plt.scatter(outDf.HCoare, outDf.HApprox, c=outDf.U10)
    plt.colorbar(label='U_10 (m/s)')
    #plt.scatter(outDf.HCoare, outDf.HApprox, c=collector_w_turb)
    #plt.colorbar(label="mean(w'T')")
    plt.xlabel("COARE")
    plt.ylabel("rho*C_P*cov(w',T')")
    plt.title("Sensible Heat Flux")
    plt.show()

    plt.plot(outDf.time, outDf.tauApprox, "-o")
    plt.plot(outDf.time, outDf.tauCoare, "-o")
    plt.legend(['EC', 'COARE'])
    plt.xlabel('Time')
    plt.ylabel('Shear Stress')
    plt.show()

    plt.plot(outDf.time, outDf.HApprox, "-o")
    plt.plot(outDf.time, outDf.HCoare, "-o")
    plt.legend(['EC', 'COARE'])
    plt.xlabel('Time')
    plt.ylabel('Sensible Heat Flux')
    plt.show()

    fig, ax = plt.subplots()
    lns1 = ax.plot(outDf.time, outDf.HApprox, "-o", label='EC')
    lns2 = ax.plot(outDf.time, outDf.HCoare, "-o", label='COARE')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sensible Heat Flux')
    ax2 = ax.twinx()
    lns3 = ax2.plot(outDf.time, outDf.U10, "-o", color='r', label='U_10')
    ax2.set_ylim([0,20])
    ax2.set_ylabel('U_10 (m/s)')
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.show()

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
        with np_load_modified(os.path.join(os.getcwd(), 'REMS', f'meteo_{cyclone}.npz')) as metFile:
            timemet = metFile['timemet.npy'] # YYYYMMDD and milliseconds past midnight
            press = metFile['press.npy'] # Barometric Pressure (hPa=mbar)
            rh = metFile['rh.npy'] # Relative Humidity (%)
            spech = metFile['spech.npy'] # Specific humidity (rh: ratio, p: Pa; T: Kelvin)
            ta = metFile['ta.npy'] # Air Temperature (C)
            solrad = metFile['solrad.npy'] # Downward Solar radiation (Wm^-2)
        with np_load_modified(os.path.join(os.getcwd(), 'REMS', f'meteo_{cyclone}_currents.npz')) as metFile:
            #timemet = metFile['timemet.npy'] # YYYYMMDD and milliseconds past midnight
            cur_n_comp = metFile['cur_n_comp.npy'] # Northward component of current velocity (m/s)
            cur_e_comp = metFile['cur_e_comp.npy'] # Eastward component of current velocity (m/s)
            tsea = metFile['tsea.npy'] # Water temperature (degC)
            depth = metFile['depth.npy'] # Approx. distance from surface (m), Babanin et al.

    remsDf = pd.DataFrame({"timemet": timemet, "press": press, "rh": rh, "spech": spech, "ta": ta, "solrad": solrad,
                            "cur_n_comp": cur_n_comp, "cur_e_comp": cur_e_comp, "tsea": tsea, "depth": depth})

    # Grabbing ERA5 data
    with np_load_modified(os.path.join(os.getcwd(), 'ERA5', 'ERA5_2015.npz')) as eraFile:
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

    eraDf = pd.DataFrame({"timemet": timemet, "u_10": u_10, "v_10": v_10, "tsea": tsea, "waveDir": waveDir, 
                            "ta": ta, "rh": rh, "spech": spech, "press": press, "solrad": solrad, "thermrad": thermrad})

    '''
    plt.plot(remsDf.timemet, remsDf.press)
    plt.plot(eraDf.timemet, eraDf.press)
    plt.xlabel('time')
    plt.ylabel('Pressure (mBar)')
    plt.legend(['REMS', 'ERA5'])
    plt.show()

    plt.plot(remsDf.timemet, remsDf.ta)
    plt.plot(eraDf.timemet, eraDf.ta)
    plt.xlabel('time')
    plt.ylabel('Sea Surface Temperature (degC)')
    plt.legend(['REMS', 'ERA5'])
    plt.show()

    plt.plot(remsDf.timemet, remsDf.tsea)
    plt.plot(eraDf.timemet, eraDf.tsea)
    plt.xlabel('time')
    plt.ylabel('Air Temperature (degC)')
    plt.legend([f'REMS (28m)', 'ERA5 (2m)'])
    plt.show()

    plt.subplot(1, 2, 1)
    plt.plot(remsDf.timemet, remsDf.tsea)
    plt.plot(remsDf.timemet, remsDf.ta)
    plt.ylim([20,30])
    plt.xlabel('time')
    plt.ylabel('Temperature (degC)')
    plt.legend([f'Sea Temp', 'Air Temp'])
    plt.title('REMS')
    plt.subplot(1,2,2)
    rems_time_period = eraDf.loc[(eraDf.timemet >= remsDf.timemet[0]) & (eraDf.timemet <= remsDf.timemet[len(remsDf) - 1])]
    plt.plot(rems_time_period.timemet, rems_time_period.tsea)
    plt.plot(rems_time_period.timemet, rems_time_period.ta)
    plt.ylim([20,30])
    plt.xlabel('time')
    plt.ylabel('Temperature (degC)')
    plt.legend([f'Sea Temp', 'Air Temp'])
    plt.title('ERA5')
    plt.show()

    time_j = []
    solrad_j = []
    time_delta = remsDf.timemet[len(remsDf) - 1] - remsDf.timemet[0]
    amountOfSlices = time_delta.total_seconds()//3600 #seconds -> hours
    solSlices = np.array_split(remsDf, amountOfSlices)
    #Integrating over each hour
    for slice in solSlices:
        slice = slice.reset_index()
        xVals = slice.timemet - slice.timemet[0]
        xVals = xVals.apply(lambda x: x.total_seconds()).values
        solrad_j.append(integrate.trapezoid(slice.solrad, x=xVals))
        time_j.append(slice.timemet[len(slice) - 1])

    plt.plot(time_j, solrad_j, "-o")
    plt.plot(eraDf.timemet, eraDf.solrad, "-o")
    plt.xlabel('time')
    plt.ylabel('Downward Solar Radiation (J/m^2)')
    plt.legend(['REMS', 'ERA5'])
    plt.show()

    time_j = []
    thermrad_j = []
    time_delta = remsDf.timemet[len(remsDf) - 1] - remsDf.timemet[0]
    amountOfSlices = time_delta.total_seconds()//3600 #seconds -> hours
    solSlices = np.array_split(remsDf, amountOfSlices)
    #Integrating over each hour
    for slice in solSlices:
        slice = slice.reset_index()
        xVals = slice.timemet - slice.timemet[0]
        xVals = xVals.apply(lambda x: x.total_seconds()).values
        thermrad_j.append(integrate.trapezoid(370*np.ones((len(xVals))), x=xVals))
        time_j.append(slice.timemet[len(slice) - 1])

    plt.plot(time_j, thermrad_j)
    plt.plot(eraDf.timemet, eraDf.thermrad)
    plt.xlabel('time')
    plt.ylabel('Downward IR Radiation (J/m^2)')
    plt.legend(['Default Value', 'ERA5'])
    plt.show()
    '''
    
    # TODO: PATCH FIX
    eraDf.solrad = eraDf.solrad/3600
    eraDf.thermrad = eraDf.thermrad/3600

    '''
    dt = eraDf.timemet.diff()
    dt = dt.apply(lambda t: t.total_seconds())
    solrad_diff = eraDf.solrad.diff()/dt

    plt.plot(remsDf.timemet, remsDf.solrad, "-o")
    #plt.plot(eraDf.timemet, solrad_diff, "-o")
    plt.plot(eraDf.timemet, eraDf.solrad, "-o")
    plt.xlabel('time')
    plt.ylabel('Downward Solar Radiation (W/m^2)')
    plt.legend(['REMS', 'ERA5'])
    plt.show()

    # NOTE: Missing plots: water current speeds

    plt.plot(remsDf.timemet, remsDf.rh)
    plt.plot(eraDf.timemet, eraDf.rh)
    #plt.plot(eraDf.timemet, 100*np.ones(len(eraDf)))
    plt.xlabel('time')
    plt.ylabel('Relative humidity (%)')
    plt.legend(['REMS','ERA5'])
    plt.show()

    plt.plot(remsDf.timemet, remsDf.spech)
    plt.plot(eraDf.timemet, eraDf.spech)
    plt.xlabel('time')
    plt.ylabel('Specific humidity (kg/kg)')
    plt.legend(['REMS','ERA5'])
    plt.show()
    '''

    #outDf = analysis_loop(readDir, remsDf, eraDf, supervised=False, cpuFraction=1, era_only=True, no_era=False)
    #outDf2 = analysis_loop(readDir, remsDf, eraDf, supervised=False, cpuFraction=1, era_only=True, no_era=False)
    #outDf2 = outDf2.loc[outDf2.HCoare != 0]

    t0 = time.perf_counter()
    write_message(f"Starting Analysis Run", filename='analysis_log.txt', writemode='w')
    for i, _ in enumerate(args.read_dir):
        readDir = Path(args.read_dir[i])
        writeDir = Path(args.write_dir[i])
        outDf = analysis_loop(readDir, remsDf, eraDf, supervised=args.run_supervised, cpuFraction=args.cpu_fraction, era_only=args.era_only, no_era=args.no_era)
        postprocess()
    t1 = time.perf_counter()
    
    write_message(f"Took {(t1-t0)/60}min", filename='analysis_log.txt')