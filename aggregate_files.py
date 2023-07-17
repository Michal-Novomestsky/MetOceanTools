import os
from pathlib import Path
from Modules.DataCleaner import DataCleaner
from Modules.DataAnalyser import write_message
import pandas as pd

def read_loop(readDir: Path) -> list:
    '''
    Checks every file and returns a list of the irregular ones.

    :param readDir: (Path) Path to the csvs/txts.
    :return: list[Path] A list of Paths for every irregular case.
    '''
    file_selector = lambda file: "csv" in file.suffix or "txt" in file.suffix
    files = [file for file in readDir.iterdir() if file_selector(file)]

    badFiles = []
    for file in files:
        fileName = file.stem
        if fileName[2:6] != "0000":
            badFiles.append(file)

    return badFiles

def combine_bad_files(badFiles: list, writeDir: os.PathLike) -> None:
    '''
    Aggregates all of the irregular files and turns them into a single csv.

    :param badFiles: (list[Path]) A list of Paths for every irregular case.
    :param writeDir: (os.PathLike) Path to the output directory for the aggregated files.
    '''
    while len(badFiles) != 0:
        file = badFiles.pop(0)
        df = DataCleaner(file).df
        fileName = file.stem
        fileDate = fileName.split('_')[1]
        fileHour = fileName[-10:-8]

        usedFiles = []
        for i, file in enumerate(badFiles):
            if (fileName.split('_')[1] == fileDate) & (fileName[-10:-8] == fileHour):
                df2 = DataCleaner(file).df
                frames = [df, df2]
                df = pd.concat(frames).reset_index()
                df = df.drop('index', axis=1)
                usedFiles.append(i)
        
        # We can only pop outside of the previous for loop to prevent breaking its order
        for i, index in enumerate(usedFiles):
            index -= i # Correcting for the change in list size from popping

            # Deleting bad files
            os.remove(badFiles[index])
            badFiles.pop(index)
        
        # Checking that there isn't already a 0000 file at the given date
        aggregated_file = Path(os.path.join(writeDir, fileName[:-8], '0000.txt'))
        if os.path.isfile(aggregated_file):
            raise FileExistsError(f'{aggregated_file} already exists.')
        
        df.to_csv(path_or_buf=aggregated_file, sep="	")

def run_aggregate_files(readDir: Path) -> None:
    '''
    Aggregates files with hours that don't start at 0000 (e.g. 0050 and 0150).

    :param readDir: (Path) Path to the csv/txt files.
    '''
    badFiles = read_loop(readDir)

    if len(badFiles) == 0:
        write_message("NO FILES WITH IRREGULAR HOUR START TIMES.", filename='cleanup_log.txt')

    else:
        write_message("FILES WITH IRREGULAR HOUR START TIMES:", filename='cleanup_log.txt')
        for file in badFiles:
            print(file.stem)
        write_message(f"{len(badFiles)} files.")

    combine_bad_files(badFiles, readDir)


if __name__=='__main__':
    readDir = Path(os.path.join(os.getcwd()), 'Rawdata', 'Nov2015')
    run_aggregate_files(readDir)