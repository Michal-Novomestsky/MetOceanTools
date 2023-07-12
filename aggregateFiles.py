import os
from pathlib import Path
import glob
import DataCleaner as dc
import pandas as pd

def read_loop(readDir: str) -> list:
    readDir += "\\*.txt"
    files = glob.glob(readDir)

    badFiles = []
    for file in files:
        fileName = file.split('_')
        if fileName[2][2:6] != "0000":
            badFiles.append(file)

    return badFiles

def combine_bad_files(badFiles: list, writeDir: str) -> None:
    while len(badFiles) != 0:
        file = badFiles.pop(0)
        df = dc.DataCleaner(file).df
        fileName = file.split('\\')[-1]
        fileDate = file.split('_')[1]
        fileHour = fileName[-10:-8]

        usedFiles = []
        for i, file in enumerate(badFiles):
            if (file.split('_')[1] == fileDate) & (file.split('\\')[-1][-10:-8] == fileHour):
                df2 = dc.DataCleaner(file).df
                frames = [df, df2]
                df = pd.concat(frames).reset_index()
                df = df.drop('index', axis=1)
                usedFiles.append(i)
        
        #We can only pop outside of the previous for loop to prevent breaking its order
        for i, index in enumerate(usedFiles):
            index -= i #Correcting for the change in list size from popping
            badFiles.pop(index)
        
        df.to_csv(path_or_buf=writeDir + '\\' + fileName[:-8] + "0000.txt", sep="	")


if __name__=='__main__':
    ###NOTE: I/O DIRECTORIES. CHANGE AS REQUIRED
    dir = os.getcwd()
    dir = str(Path(dir).parents[0])
    readDir = dir + "\\Rawdata\\Nov2015"
    writeDir = dir + "\\Aggregates\\Nov2015_aggregates"

    badFiles = read_loop(readDir)

    print("BAD FILES:")
    for file in badFiles:
        fileName = file.split('\\')[-1]
        print(fileName)

    combine_bad_files(badFiles, writeDir)