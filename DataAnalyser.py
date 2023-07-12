import pandas as pd
import numpy as np
import numpy.typing as npt

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