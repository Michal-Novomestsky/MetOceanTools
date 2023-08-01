import os
import multiprocessing as mp
import numpy as np
import time
import argparse

from Modules.DataCleaner import DataCleaner
from Modules.DataAnalyser import write_message
from aggregate_files import run_aggregate_files
from pathlib import Path


def cleanup_loop(readDir: Path, writeDir: Path, supervised=False, cpuFraction=1, file_selector_name=None) -> None:
    """
    Steps through each data file located in readDir and outputs a cleaned-up version in writeDir. Otherwise keeps track of all rejected files.
    supervised sets whether user verification is needed to accept changes or whether it is completed automatically. Unsupervised enables multiprocessing
    using cpuFraction% of all available cores.
    """
    rejectedFiles = []
    if file_selector_name is None:
        file_selector=lambda file: "csv" in file.suffix or "txt" in file.suffix
    else:
        file_selector=lambda file: ("csv" in file.suffix or "txt" in file.suffix) and (file_selector_name in file.stem)
    files = [file for file in readDir.iterdir() if file_selector(file)]

    # Making folders
    os.mkdir(os.path.join(writeDir, 'FTs'))
    os.mkdir(os.path.join(writeDir, 'FTs', 'loglogs'))
    os.mkdir(os.path.join(writeDir, 'hists'))

    # Manual one-by-one checking
    if supervised:
        write_message('Starting Supervised Run', filename='cleanup_log.txt')
        for file in files:
            rejectedFile = _cleanup_iteration(file, writeDir, supervised=True)
            if rejectedFile is not None:
                rejectedFiles.append(rejectedFile)

    # Enabling multiprocessing >:)
    else:
        if cpuFraction > 1 or cpuFraction <= 0:
            raise ValueError("cpuFraction must be within (0,1]")

        cpuCount = mp.cpu_count()
        coresToUse = int(np.ceil(cpuFraction*cpuCount))
        write_message(f"Using {100*cpuFraction}% of available cores -> {coresToUse}/{cpuCount}", filename='cleanup_log.txt')

        # Creating a tuple of tuples of inputs to pass into each iteration
        writeDirArr = [writeDir]*len(files)
        supervisedArr = [supervised]*len(files)
        args = [*zip(files, writeDirArr, supervisedArr)]

        with mp.Pool(coresToUse) as p:
            rejectedFiles = p.starmap(_cleanup_iteration, iterable=args)

    write_message("Cleanup run done!", filename='cleanup_log.txt')
    write_message(f"Rejected files:", filename='cleanup_log.txt')
    rejectedFileCount = 0
    for file in rejectedFiles:
        if file is not None:
            write_message(file, filename='cleanup_log.txt')
            rejectedFileCount += 1
    write_message(f"{rejectedFileCount} files rejected", filename='cleanup_log.txt')

def _cleanup_iteration(file: Path, writeDir: Path, supervised=True) -> str:
    """
    Internal function which runs an iteration of a cleanup run. Iterated externally by cleanup_loop.
    """
    data = DataCleaner(file)

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
        # Interpolating points in comp and MRU to bring it up to the same resolution
        data.remove_nans(comp1, data.df, naive=True)
        data.remove_nans(comp2, data.df, naive=True)
        data.remove_nans(mru_pitch, data.df, naive=True)
        data.remove_nans(mru_yaw, data.df, naive=True)
        data.remove_nans(mru_roll, data.df, naive=True)
        data.remove_nans(mru_p, data.df, naive=True)
        data.remove_nans(mru_r, data.df, naive=True)
        data.remove_nans(mru_y, data.df, naive=True)
        write_message(f"{fileName}: Interpolated", filename='cleanup_log.txt')

        # Motion correction
        data.mru_correct()
        write_message(f"{fileName}: Motion Corrected", filename='cleanup_log.txt')

        # Pruning
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
        write_message(f"{fileName}: Pruned", filename='cleanup_log.txt')
        
        # FFT plotting/checking
        # The if nots are there as a simplistic means of lazychecking to prevent unecessary computation if we've already hit a faulty dataset
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
        # Not filtering with temperature FTs since their regression is poorly studied
        if not rejectLog:
            data.plot_ft_loglog(t1, fileName, gradient=-1, gradient_cutoff=100, pearson_cutoff=0, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)
        if not rejectLog:
            data.plot_ft_loglog(t2, fileName, gradient=-1, gradient_cutoff=100, pearson_cutoff=0, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2)

        # Hist plotting/checking
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
            rejectLog = rejectLog or data.reject_file_on_range(t1, margain=3, sec_stepsize=10*60)
        # if not rejectLog:
            # rejectLog = rejectLog or data.reject_file_on_range(t2, margain=3, sec_stepsize=10*60)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(t1, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=300)
        if not rejectLog:
            rejectLog = rejectLog or data.plot_hist(t2, fileName, diffCutOff=8, supervised=supervised, saveLoc=saveLoc, bins=300)

        # Plotting points which were removed
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
            
        write_message(f"{fileName}: Plotting/Sanity Checking Complete", filename='cleanup_log.txt')

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
        write_message(f"Rejected {fileName}: Recursion error", filename='cleanup_log.txt')
        rejectLog = True

    if supervised:
        # Writing cleaned up file or rejecting it
        inputLoop = True
        while inputLoop:
            isAcceptable = input("We happy? [Y/N] ")
            if isAcceptable.lower() == 'y':
                data.df.to_csv(path_or_buf=writeDir + fileName, sep="	")
                write_message("Yeah. We happy", filename='cleanup_log.txt')
                inputLoop = False

            elif isAcceptable.lower() == "n":
                write_message(f"Rejected {fileName}", filename='cleanup_log.txt')
                return fileName
            
            else:
                write_message("Invalid input. Try again.", filename='cleanup_log.txt')

    # If unsupervised, auto-write every time
    else:
        # Catching faulty datasets
        if rejectLog:
            write_message(f"REJECTED: {fileName}", filename='cleanup_log.txt')
            return fileName
        else:
            data.df.to_csv(path_or_buf=os.path.join(writeDir, fileName), sep="	")
            write_message(f"Cleaned up {fileName}", filename='cleanup_log.txt')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', nargs='+', type=str, help='Path to the rawdata. Can be a list.')
    parser.add_argument('--write_dir', nargs='+', type=str, help='Path to output. Can be a list.')
    parser.add_argument('--cpu_fraction', type=float, help='% Of CPUs to use. Can be within (0,1].', default=1)
    parser.add_argument('--run_supervised', action='store_true', help='Run one-by-one cleanup.', default=False)
    parser.add_argument('--file_selector_name', type=str, help='String which appears in the intended filename.', default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    write_message("Starting Cleanup Run", filename='cleanup_log.txt', writemode='w')
    for i, _ in enumerate(args.read_dir):
        readDir = Path(args.read_dir[i])
        writeDir = Path(args.write_dir[i])

        cleanup_loop(readDir, writeDir, supervised=args.run_supervised, cpuFraction=args.cpu_fraction, file_selector_name=args.file_selector_name)
        run_aggregate_files(writeDir)
    t1 = time.perf_counter()
    
    write_message(f"Took {(t1-t0)/3600}hrs", filename='cleanup_log.txt')