import os
import multiprocessing as mp
import numpy as np
import time
import argparse

from Modules.DataCleaner import DataCleaner
from Modules.DataAnalyser import write_message
from aggregate_files import run_aggregate_files
from pathlib import Path

TIME_INTERVAL = 20 # TODO Voermans et al. set to 15min

def cleanup_loop(readDir: Path, writeDir: Path, supervised=False, cpuFraction=1, file_selector_name=None, mru_correct=True, generate_plots=True) -> None:
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
    os.mkdir(os.path.join(writeDir, 'comparisons'))

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
        mruArr = [mru_correct]*len(files)
        plotArr = [generate_plots]*len(files)
        args = [*zip(files, writeDirArr, supervisedArr, mruArr, plotArr)]

        with mp.Pool(coresToUse) as p:
            rejectedFiles = p.starmap(_cleanup_iteration, iterable=args)

    write_message("Cleanup run done!\nRejected files:", filename='cleanup_log.txt')
    rejectedFileCount = 0
    for file in rejectedFiles:
        if file is not None:
            write_message(file, filename='cleanup_log.txt')
            rejectedFileCount += 1
    write_message(f"{rejectedFileCount} files rejected", filename='cleanup_log.txt')

def _cleanup_iteration(file: Path, writeDir: Path, supervised=True, mru_correct=True, generate_plots=True) -> str:
    """
    Internal function which runs an iteration of a cleanup run. Iterated externally by cleanup_loop.
    """
    data = DataCleaner(file)

    fileName = file.stem

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
    
    # Interpolating points in comp and MRU to bring it up to the same resolution
    for entry in [comp1, comp2, mru_pitch, mru_yaw, mru_roll, mru_p, mru_r, mru_y]:
        data.remove_nans(entry, data.df, naive=True)
    write_message(f"{fileName}: Interpolated", filename='cleanup_log.txt')

    # Motion correction
    if mru_correct:
        data.mru_correct()
        write_message(f"{fileName}: Motion Corrected", filename='cleanup_log.txt')
    else:
        write_message(f"{fileName}: MRU CORRECTION OFF", filename='cleanup_log.txt')

    # Pruning
    for entry in [u1, u2, v1, v2, w1, w2, t1, t2]:
        data.prune_or([data.gradient_cutoff(entry, 3)])
        data.prune_or([data.std_cutoff(entry, 5, sec_stepsize=TIME_INTERVAL*60)])
    write_message(f"{fileName}: Pruned", filename='cleanup_log.txt')
    
    # All subsequent analyses are skipped if an erroneous parameter is idenfied earlier with rejectLog
    # FFT plotting/checking
    saveLoc = os.path.join(writeDir, "FTs", "loglogs")
    rejectLog = False
    for entry in [u1, u2, v1, v2, w1, w2]:
        if rejectLog:
            break
        rejectLog = rejectLog or data.plot_ft_loglog(entry, fileName, gradient=-5/3, gradient_cutoff=0.5, pearson_cutoff=0.8, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=TIME_INTERVAL, windowWidth=2, generate_plots=generate_plots)

    for t in [t1, t2]:
        # Not filtering with temperature FTs since their regression is poorly studied
        if rejectLog:
            break
        data.plot_ft_loglog(t, fileName, gradient=-1, gradient_cutoff=100, pearson_cutoff=0, supervised=supervised, saveLoc=saveLoc, plotType="-", turbSampleMins=20, windowWidth=2, generate_plots=generate_plots)

    # Hist checking
    for entry in [u1, u2, v1, v2, w1, w2, t1, t2]:
        if rejectLog:
            break
        rejectLog = rejectLog or data.reject_hist_outliers(entry, diffCutoff=8)
        # Seperate if-statement to prevent printing from rejectLogs caused by prior filter passes (e.g. plot_ft_loglog)
        if rejectLog:
            print(f"Rejected {fileName}: Histogram has a spike")

    if not rejectLog:
        # Checking if temperature has an unusually large range or is mean shifting
        is_temp_fluctuating = data.reject_file_on_changing_mean(t1, margain=4, sec_stepsize=10*60, n_most=1)
        data.df.is_temp1_fluctuating = len(data.df)*[is_temp_fluctuating]

        is_temp_range_large = data.range_cutoff(t1, margain=2.5, sec_stepsize=5*60)
        data.df.is_temp1_range_large = is_temp_range_large

        is_temp_fluctuating = data.reject_file_on_changing_mean(t2, margain=4, sec_stepsize=10*60, n_most=1)
        data.df.is_temp2_fluctuating = len(data.df)*[is_temp_fluctuating]

        is_temp_range_large = data.range_cutoff(t2, margain=2.5, sec_stepsize=5*60)
        data.df.is_temp2_range_large = is_temp_range_large

        # Generating plots
        if generate_plots:
            # Hists
            saveLoc = os.path.join(writeDir, "hists")
            for entry in [u1, u2, v1, v2, w1, w2]:
                data.plot_hist(entry, fileName, supervised=supervised, saveLoc=saveLoc, bins=1000)
            for t in [t1, t2]:
                data.plot_hist(t, fileName, supervised=supervised, saveLoc=saveLoc, bins=300)

            # Plotting original timeseries vs filtered ones
            saveLoc = os.path.join(writeDir, "comparisons")
            for entry in [u1, u2, v1, v2, w1, w2]:
                data.plot_comparison(entry, fileName, supervised=supervised, saveLoc=saveLoc)
            for t in [t1, t2]:
                data.plot_comparison(t, fileName, supervised=supervised, saveLoc=saveLoc, y_lim=[15, 40])

    write_message(f"{fileName}: Plotting/Sanity Checking Complete", filename='cleanup_log.txt')

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
    parser.add_argument('--mru_correct', action='store_true', help='Correct for MRU inclination.', default=False)
    parser.add_argument('--generate_plots', action='store_true', help='Save plots.', default=False)
    parser.add_argument('--file_selector_name', type=str, help='String which appears in the intended filename.', default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    write_message("Starting Cleanup Run", filename='cleanup_log.txt', writemode='w')
    for i, _ in enumerate(args.read_dir):
        readDir = Path(args.read_dir[i])
        writeDir = Path(args.write_dir[i])

        cleanup_loop(readDir, writeDir, supervised=args.run_supervised, cpuFraction=args.cpu_fraction, 
                     file_selector_name=args.file_selector_name, mru_correct=args.mru_correct,
                     generate_plots=args.generate_plots)
        run_aggregate_files(writeDir)
    t1 = time.perf_counter()
    
    write_message(f"Took {round((t1-t0)/3600, 1)}hrs", filename='cleanup_log.txt')