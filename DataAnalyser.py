import pandas as pd
import os
import numpy.typing as npt

def write_message(message: str, filename: os.PathLike, writemode='a'):
    """
    Writes a message to both the terminal and the output file at dir.

    :param message: (str) Message to print.
    :param dir: (os.PathLike) Path to the output file.
    :param writemode: (str) Default IO write method (e.g. 'a' for append, 'w' for write, etc.).
    """
    dir = os.path.join(os.cwd(), filename)

    print(message)
    with open(dir, writemode) as f:
        f.write(message + '\n')

class DataAnalyser:
    def __init__(self, dir) -> None:
        try:
            self.df = pd.read_csv(dir, sep = "	")
        except:
            raise ValueError('Invalid csv file (may either be file type or its contents)')

        #Handy values
        self.mean = self.df.mean()
        self.std = self.df.std()

    def insert_col(self, index: int, title: str, new_column: npt.ArrayLike) -> None:
        """
        Inserts a column new_column into the dataframe at the position index, with a column header title
        """
        if type(index) != int:
            raise TypeError("Index must be int")
        if len(new_column) != len(self.df):
            raise ValueError('Dataframe and the new column to be added must be of the same length')

        self.df.insert(index, title, new_column)