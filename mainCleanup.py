import os
import DataCleaner as dc
from pathlib import Path
import glob
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import time

def cleanup_loop(readDir: Path, writeDir: Path, supervised=True, cpuFraction=75):
    """
    Steps through each data file located in readDir and outputs a cleaned-up version in writeDir. Otherwise keeps track of all rejected files.
    supervised sets whether user verification is needed to accept changes or whether it is completed automatically. Unsupervised enables multiprocessing
    using cpuFraction% of all available cores.
    """
    test = readDir
    readDir += "\\*.txt"
    writeDir += "\\"
    rejectedFiles = []
    files = glob.glob(readDir)

    #Manual one-by-one checking
    if supervised:
        for file in files:
            #file = test + "\\NRAFBR_20042015_010000.txt"
            print('Starting Supervised Run')
            rejectedFile = _cleanup_iteration(file, writeDir, supervised=True)
            if rejectedFile is not None:
                rejectedFiles.append(rejectedFile)

    #Enabling multiprocessing >:)
    else:
        if cpuFraction > 100 or cpuFraction <= 0:
            raise ValueError("cpuFraction must be between 1-100%")

        cpuCount = mp.cpu_count()
        coresToUse = int(np.ceil((cpuFraction/100)*cpuCount))
        print(f"Using {cpuFraction}% of available cores -> {coresToUse}/{cpuCount}")

        #Creating a tuple of tuples of inputs to pass into each iteration
        writeDirArr = [writeDir]*len(files)
        supervisedArr = [supervised]*len(files)
        args = [*zip(files, writeDirArr, supervisedArr)]

        with mp.Pool(coresToUse) as p:
            rejectedFiles = p.starmap(_cleanup_iteration, iterable=args)

    print("Cleanup run done!")
    print(f"Rejected files:")
    rejectedFileCount = 0
    for file in rejectedFiles:
        if file is not None:
            print(file)
            rejectedFileCount += 1
    print(f"{rejectedFileCount} files rejected")

def _cleanup_iteration(file: os.PathLike, writeDir: Path, supervised=True) -> str|None:
    """
    Internal function which runs an iteration of a cleanup run. Iterated externally by cleanup_loop.
    """
    file = Path(file)
    data = dc.DataCleaner(file)

    fileName = file.stem

    ###NOTE: EDIT THIS TO GRAB THE DATAPOINTS YOU NEED FOR A PARTICULAR CLEANUP RUN
    w1 = "Anemometer #1 W Velocity (ms-1)"
    w2 = "Anemometer #2 W Velocity (ms-1)"
    u1 = "Anemometer #1 U Velocity (ms-1)"
    u2 = "Anemometer #2 U Velocity (ms-1)"
    v1 = "Anemometer #1 V Velocity (ms-1)"
    v2 = "Anemometer #2 V Velocity (ms-1)"
    t1 = "Anemometer #1 Temperature (degC)"
    t2 = "Anemometer #2 Temperature (degC)"
    comp1 = "Compass #1 (deg)"
    comp2 = "Compass #2 (deg)"
    mru_pitch = 'MRU Pitch Angle (deg)'
    mru_yaw = 'MRU Yaw Angle (deg)'
    mru_roll = 'MRU Roll Angle (deg)'
    mru_p = 'MRU P Axis Velocity'
    mru_r = 'MRU R Axis Velocity'
    mru_y = 'MRU Y Axis Velocity'
    
    try:
        #Interpolating points in comp and MRU to bring it up to the same resolution
        data.remove_nans(comp1, data.df, naive=True)
        data.remove_nans(comp2, data.df, naive=True)
        data.remove_nans(mru_pitch, data.df, naive=True)
        data.remove_nans(mru_yaw, data.df, naive=True)
        data.remove_nans(mru_roll, data.df, naive=True)
        data.remove_nans(mru_p, data.df, naive=True)
        data.remove_nans(mru_r, data.df, naive=True)
        data.remove_nans(mru_y, data.df, naive=True)
        print(f"{fileName}: Interpolated")

        #Motion correction
        data.mru_correct()
        print(f"{fileName}: Motion Corrected")

        #Pruning
        data.remove_nans(w1, data.originalDf)
        data.prune_or(w1, (data.std_cutoff(w1, 8), data.gradient_cutoff(w1, 2)))
        data.prune_or(w1, (data.std_cutoff(w1, 8), data.gradient_cutoff(w1, 2)))

        data.remove_nans(w2, data.originalDf)
        data.prune_or(w2, (data.std_cutoff(w2, 8), data.gradient_cutoff(w2, 1.5)))
        data.prune_or(w2, (data.std_cutoff(w2, 8), data.gradient_cutoff(w2, 2)))

        data.remove_nans(u1, data.originalDf)
        data.prune_or(u1, (data.std_cutoff(u1, 8), data.gradient_cutoff(u1, 2)))
        data.prune_or(u1, (data.std_cutoff(u1, 8), data.gradient_cutoff(u1, 2)))

        data.remove_nans(u2, data.originalDf)
        data.prune_or(u2, (data.std_cutoff(u2, 8), data.gradient_cutoff(u2, 2)))
        data.prune_or(u2, (data.std_cutoff(u2, 8), data.gradient_cutoff(u2, 2)))

        data.remove_nans(v1, data.originalDf)
        data.prune_or(v1, (data.std_cutoff(v1, 8), data.gradient_cutoff(v1, 2)))
        data.prune_or(v1, (data.std_cutoff(v1, 8), data.gradient_cutoff(v1, 2)))

        data.remove_nans(v2, data.originalDf)
        data.prune_or(v2, (data.std_cutoff(v2, 8), data.gradient_cutoff(v2, 1.5)))
        data.prune_or(v2, (data.std_cutoff(v2, 8), data.gradient_cutoff(v2, 2)))

        data.remove_nans(t1, data.originalDf)
        data.prune_and(t1, (data.std_cutoff(t1, 6), data.gradient_cutoff(t1, 1.5)))
        data.prune_and(t1, (data.std_cutoff(t1, 6), data.gradient_cutoff(t1, 2)))

        data.remove_nans(t2, data.originalDf)
        data.prune_and(t2, (data.std_cutoff(t2, 6), data.gradient_cutoff(t2, 1.5)))
        data.prune_and(t2, (data.std_cutoff(t2, 6), data.gradient_cutoff(t2, 2)))
        print(f"{fileName}: Pruned")
        
        #FFT plotting/checking
        #The if nots are there as a simplistic means of lazychecking to prevent unecessary computation if we've already hit a faulty dataset
        saveLoc = os.path.join(writeDir, "FTs", "loglogs")
        rejectLog = data.plot_ft_loglog(w1, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_ft_loglog(w2, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_ft_loglog(u1, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_ft_loglog(u2, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_ft_loglog(v1, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_ft_loglog(v2, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        #Not filtering with temperature FTs since their regression is poorly studied
        if not rejectLog:
            data.plot_ft_loglog(t1, fileName, gradient=-1, gradient_cutoff=100, pearson_cutoff=0, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            data.plot_ft_loglog(t2, fileName, gradient=-1, gradient_cutoff=100, pearson_cutoff=0, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)

        #Hist plotting/checking
        saveLoc = os.path.join(writeDir, "hists")
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(w1, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(w2, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(u1, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(u2, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(v1, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(v2, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=1000)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(t1, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=300)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(t2, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=300)

        #Plotting points which were removed
        saveLoc = os.path.join(writeDir, "hists")
        if not rejectLog:
            data.plot_comparison(w1, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(w2, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(u1, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(u2, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(v1, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(v2, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(t1, fileName, supervised=supervised, saveLoc=saveLoc)
            data.plot_comparison(t2, fileName, supervised=supervised, saveLoc=saveLoc)
            
        print(f"{fileName}: Plotting/Sanity Checking Complete")

        '''
        w1 = "Anemometer #1 W Velocity (ms-1)"
        data.remove_nans(w1, data.originalDf)
        data.prune_or(w1, (data.std_cutoff(w1, 3), data.gradient_cutoff(w1, 2)))

        w2 = "Anemometer #2 W Velocity (ms-1)"
        data.remove_nans(w2, data.originalDf)
        data.prune_or(w2, (data.std_cutoff(w2, 3), data.gradient_cutoff(w2, 2)))
        data.prune_or(w2, (data.std_cutoff(w2, 3), data.gradient_cutoff(w2, 2)))

        u1 = "Anemometer #1 U Velocity (ms-1)"
        data.remove_nans(u1, data.originalDf)
        data.prune_or(u1, (data.std_cutoff(u1, 3), data.gradient_cutoff(u1, 2)))

        u2 = "Anemometer #2 U Velocity (ms-1)"
        data.remove_nans(u2, data.originalDf)
        data.prune_or(u2, (data.std_cutoff(u2, 3), data.gradient_cutoff(u2, 2)))
        data.prune_or(u2, (data.std_cutoff(u2, 3), data.gradient_cutoff(u2, 2)))

        v1 = "Anemometer #1 V Velocity (ms-1)"
        data.remove_nans(v1, data.originalDf)
        data.prune_or(v1, (data.std_cutoff(v1, 3), data.gradient_cutoff(v1, 2)))

        v2 = "Anemometer #2 V Velocity (ms-1)"
        data.remove_nans(v2, data.originalDf)
        data.prune_or(v2, (data.std_cutoff(v2, 3), data.gradient_cutoff(v2, 2)))
        data.prune_or(v2, (data.std_cutoff(v2, 3), data.gradient_cutoff(v2, 2)))

        t1 = "Anemometer #1 Temperature (degC)"
        data.remove_nans(t1, data.originalDf)
        data.prune_and(t1, (data.std_cutoff(t1, 3), data.gradient_cutoff(t1, 2)))
        data.prune_and(t1, (data.std_cutoff(t1, 3), data.gradient_cutoff(t1, 2)))

        t2 = "Anemometer #2 Temperature (degC)"
        data.remove_nans(t2, data.originalDf)
        data.prune_and(t2, (data.std_cutoff(t2, 3), data.gradient_cutoff(t2, 2)))
        data.prune_and(t2, (data.std_cutoff(t2, 3), data.gradient_cutoff(t2, 2)))

        data.plot_comparison(w1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(w2, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(u1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(u2, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(v1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(v2, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(t1, fileName, supervised=supervised, saveLoc=writeDir + "plots")
        data.plot_comparison(t2, fileName, supervised=supervised, saveLoc=writeDir + "plots")

        #data.plot_ft_dev_loglog(w1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(w2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(u1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(u2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(v1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(v2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(t1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        #data.plot_ft_dev_loglog(t2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\devs", plotType="-")
        
        data.plot_ft_loglog(w1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(w2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(u1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(u2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(v1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(v2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-5/3, windowWidth=2)
        data.plot_ft_loglog(t1, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-1, windowWidth=2)
        data.plot_ft_loglog(t2, fileName, supervised=supervised, saveLoc=writeDir + "FTs\\loglogs", plotType="-", turbSampleMins=20, gradient=-1, windowWidth=2)
        
        data.plot_hist(w1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(w2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(u1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(u2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(v1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(v2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=1000)
        data.plot_hist(t1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=200)
        data.plot_hist(t2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=200)
        
        comp1 = "Compass #1 (deg)"
        #data.remove_nans(comp1, data.originalDf)
        data.remove_nans(comp1, data.df, naive=True)

        #comp2 = "Compass #2 (deg)"
        #data.remove_nans(comp2, data.df, naive=True)

        data.plot_hist(comp1, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=700)
        #data.plot_hist(comp2, fileName, supervised=supervised, saveLoc=writeDir + "hists", bins=700)
        
        data.remove_nans(comp1, data.df, naive=True)
        data.remove_nans(comp2, data.df, naive=True)
        data.remove_nans(mru_pitch, data.df, naive=True)
        data.remove_nans(mru_yaw, data.df, naive=True)
        data.remove_nans(mru_roll, data.df, naive=True)
        data.remove_nans(mru_p, data.df, naive=True)
        data.remove_nans(mru_r, data.df, naive=True)
        data.remove_nans(mru_y, data.df, naive=True)
        '''
    except RecursionError:
        print(f"Rejected {fileName}: Recursion error")
        rejectLog = True

    if supervised:
        #Writing cleaned up file or rejecting it
        inputLoop = True
        while inputLoop:
            isAcceptable = input("We happy? [Y/N] ")
            if isAcceptable.lower() == 'y':
                data.df.to_csv(path_or_buf=writeDir + fileName, sep="	")
                print("Yeah. We happy")
                inputLoop = False

            elif isAcceptable.lower() == "n":
                print(f"Rejected {fileName}")
                return fileName
            
            else:
                print("Invalid input. Try again.")

    #If unsupervised, auto-write every time
    else:
        #Catching faulty datasets
        if rejectLog:
            print(f"REJECTED: {fileName}")
            return fileName
        else:
            data.df.to_csv(path_or_buf=os.path.join(writeDir, fileName), sep="	")
            print(f"Cleaned up {fileName}")

if __name__=='__main__':
    ###NOTE: I/O DIRECTORIES. CHANGE AS REQUIRED
    dir = os.getcwd()
    dir = str(Path(dir).parents[0])
    #readDir = dir + "\\Apr2015_clean_MRU_and_compasses"
    readDir = os.path.join(dir + "Cleanup Inputs", "Apr2015_cleanup_input")
    writeDir = dir + "Fullsweeps", "Apr2015_fullsweep"

    t0 = time.time()
    cleanup_loop(readDir, writeDir, supervised=True, cpuFraction=60)
    t1 = time.time()
    print(f"Took {t1-t0}s")

    # ioList = ['Sep2015','Nov2015']

    # for io in ioList:
    #     t0 = time.time()
    #     try:
    #         t0 = time.time()
    #         cleanup_loop(dir + "\\Cleanup Inputs\\" + io + "_cleanup_input", dir + "\\Fullsweeps\\" + io + "_fullsweep", supervised=False, cpuFraction=60)
    #     except ValueError:
    #         print('CRASHED')
    #     t1 = time.time()
    #     print(f"Took {t1-t0}s")