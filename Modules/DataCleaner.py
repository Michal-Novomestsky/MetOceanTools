import pandas as pd
import numpy as np
import os
import seaborn as sns

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import linregress

class DataCleaner:
    def __init__(self, dir: Path) -> None:
        # Reading in file
        with open(dir, "r") as file:
            lines = file.readlines()

            if lines[0] == "% NRA Flarebridge Research Data provided by Woodside Energy Ltd, collected by RPS OceanMonitor System v1.3\n":
                # Grabbing headers
                columnNames = lines[3].split("	")
                units = lines[4].split("	")

                tempColumns = []
                # We give special treatment to the first and last elements to avoid grabbing %Units and the like (these two entries don't have units anyway)
                # We also chop off parts of the string in these two cases to avoid % and \n
                tempColumns.append(columnNames[0][2:])
                for i in range(1, len(columnNames) - 1):
                    # Avoiding adding spaces if the unit is dimensionless
                    if units[i] == "":
                        tempColumns.append(columnNames[i])
                    else:
                        tempColumns.append(columnNames[i] + " (" + units[i] + ")")
                tempColumns.append(columnNames[len(columnNames) - 1][:len(columnNames[len(columnNames) - 1]) - 1])
                
                self.df = pd.read_csv(dir, sep = "	", skiprows=7)
                self.df.columns = tempColumns

                # Adding in a global second entry to avoid having to deal with modular seconds
                globSeconds = self.df.Second.add(60*self.df.Minute)
                self.df.insert(3, "GlobalSecs", globSeconds)

                self.df.insert(0, "is_temp_fluctating", len(self.df)*[False])
                self.df.insert(0, "is_temp_range_large", len(self.df)*[False])
            
            # If the file has previously been cleaned, we can just directly read from it without any tidying required
            else:
                self.df = pd.read_csv(dir, sep = "	")
                # Chopping off erroneous endpoints where time resets
                self.df = self.df.loc[self.df.index < len(self.df) - 1]

            # Keeping an unedited copy for reference
            self.originalDf = self.df.copy(deep=True)

    def plot_comparison(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_COMPARISON_", y_lim=None) -> None:
        """
        Presents a plot of the values removed from entry during cleanup in the .txt fileName. plotType specifies the plotting marker.
        """
        title = fileName[:len(fileName) - 4] + plotTitle + entry

        #sns.lineplot(data=self.originalDf, x='GlobalSecs', y=entry, label='Original', color='r')
        sns.scatterplot(data=self.originalDf, x='GlobalSecs', y=entry, color='r', edgecolor=None, marker='.')
        #sns.lineplot(data=self.df, x='GlobalSecs', y=entry, label='Cleaned', color='b')
        sns.scatterplot(data=self.df, x='GlobalSecs', y=entry, color='b', edgecolor=None, marker='.')
        plt.ylabel(entry)
        plt.title(title)

        if y_lim is not None:
            plt.ylim(y_lim)

        if supervised:
            plt.show()
        else:
            plt.savefig(os.path.join(saveLoc, f"{title}.png"))
            plt.close()

    def plot_ft_dev_loglog(self, entry: str, fileName: str, supervised=False, saveLoc=None, plotTitle="_FT_DEVIATION_", plotType="-o", sampleSpacing=1) -> None:
        """
        NOTE: CURRENTLY DEPRECIATED.
        Presents a loglog plot of the deviation in FFT as a result of the cleanup. sampleSpacing is the spacing between points in
        the frequency domain. Refer to plot_comparison for parameter details.
        """
        N = len(self.df[entry])
        ft_x = fftfreq(N, sampleSpacing)[:N//2] # The function is symmetric so we are only interested in +ive frequency values
        ft_x = ft_x/(2*np.pi)

        ft_yChanged = fft(self.df[entry].values)
        ft_yChanged = 2/N * np.abs(ft_yChanged[:N//2])**2 # Multiplying by 2 to deal with the fact that we chopped out -ive freqs
        ft_yChanged[np.abs(ft_yChanged) < 1e-2] = 1e-2 # Capping miniscule values off to prevent deviation from blowing up

        ft_yOriginal = fft(self.originalDf[entry].values)
        ft_yOriginal = 2/N * np.abs(ft_yOriginal[:N//2])**2
        ft_yOriginal[np.abs(ft_yOriginal) < 1e-2] = 1e-2
        
        dev = np.abs(ft_yChanged - ft_yOriginal)

        title = fileName[:len(fileName) - 4] + plotTitle + entry

        plt.loglog(ft_x, dev, plotType)
        plt.xlabel('Frequency Domain')
        plt.ylabel('Deviation')

        plt.title(title)

        if supervised:
            plt.show()
        else:
            plt.savefig(os.path.join(saveLoc, f"{title}.png"))
            plt.close()

    def plot_ft_loglog(self, entry: str, fileName: str, gradient, gradient_cutoff, pearson_cutoff, supervised=False, saveLoc=None, plotTitle="_FT_LOGLOG_", plotType="-o", sampleSpacing=1, turbSampleMins=None, windowWidth=None) -> None:
        """
        Presents a plot of the FFT spectral density with both x and y axes in log10. Turbulent switches on u' = u - u_bar with u_bar evaulated over 
        turbSampleMins minutes averaged over the total time and gradient provides the slope for a line in the tail of the FT curve. Does not plot and returns
        True if the magnitude of the Pearson correlation coefficient of a line of best fit is < pearson_cutoff and abs(gradient - m) > gradient_cutoff.
        Refer to plot_ft_dev for other parameter details.
        """

        '''
        if windowWidth is not None:
            N_windows = self.df.Minute[len(self.df) - 1]//windowWidth - 1
            window = signal.windows.hamming(len(self.df.loc[self.df.Minute <= windowWidth]))

            for i in range(N_windows):
                vals = timeSeries.loc[(i*windowWidth <= self.df.Minute) & (self.df.Minute <= (i + 1)*windowWidth), entry].values
                timeSeries.loc[(i*windowWidth <= self.df.Minute) & (self.df.Minute <= (i + 1)*windowWidth), entry] = vals*window
        '''
        if turbSampleMins is not None:
            if windowWidth is not None:
                N = len(self.df.loc[(0 <= self.df.Minute) & (self.df.Minute <= windowWidth)])
                #TODO: Broken x axis
                # If GlobalSecs is empty, reject the case
                try:
                    ft_x = fftfreq(N, d=self.df.GlobalSecs[1])[:N//2] #The function is symmetric so we are only interested in +ive frequency values
                except KeyError:
                    return True
                #ft_x = ft_x/(2*np.pi)

                # Loop through as many times as required to get all snapshots
                snapshots = range(self.df.Minute[len(self.df) - 1]//turbSampleMins)
                for j in snapshots:
                    tLower = j*turbSampleMins
                    tUpper = (j + 1)*turbSampleMins
                    timeSeries = self.df.loc[(tLower <= self.df.Minute) & (self.df.Minute <= tUpper)].copy(deep=True)

                    # Getting average of all time windows
                    ft_y = self._window_width_aux(N, timeSeries, windowWidth, entry, tLower, tUpper)
                    title = fileName[:len(fileName) - 4] + plotTitle + f"_{tLower}-{tUpper}MIN_" + f"_WINDOW_{windowWidth}MIN_" + entry

                    # If the fft array is empty, we simply reject the case
                    if ft_y is None:
                        return True

                    logical = ft_y > min(ft_y) # Removing erroneous minimum value which is << the second smallest
                    if j == 0: # Catch to stop ft_x from shortening each time round (it gets defined outside the for loop)
                        ft_x = ft_x[logical]
                    ft_y = ft_y[logical]

                    plt.loglog(ft_x, ft_y, plotType, color='b')
                    plt.xlabel('Frequency Domain', fontsize=7)
                    plt.ylabel(f'Spectral Density of Clean {entry} Data', fontsize=7)
                    plt.title(title, fontsize=7)

                    # Plotting the line on top
                    self._gradient_line_aux(ft_y, ft_x, gradient)

                    # Checking if the line fits certain regression conditions and plotting
                    xVal = ft_x[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y)))]
                    yVal = ft_y.loc[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y))) - 1] # -1 to stop extra point being added in
                    m, _, pearson_r, _, _ = linregress(np.log10(xVal), np.log10(yVal))
                    if supervised:
                        plt.show()
                    elif np.abs(pearson_r) < pearson_cutoff and np.abs(gradient - m) > gradient_cutoff:
                        print(f"Rejected {fileName}: Pearson R = {pearson_r}, m = {m}")
                        plt.close()
                        return True
                    else:
                        plt.savefig(os.path.join(saveLoc, f"{title}.png"))
                        plt.close()
                    
                    return False
                    
            else:
                N = len(self.df)
                ft_x = fftfreq(N, sampleSpacing)[:N//2] # The function is symmetric so we are only interested in +ive frequency values
                ft_x = ft_x/(2*np.pi)

                # Loop through as many times as required to get all snapshots
                snapshots = range(self.df.Minute[len(self.df) - 1]//turbSampleMins)
                for j in snapshots:
                    tLower = j*turbSampleMins
                    tUpper = (j + 1)*turbSampleMins
                    timeSeries = self.df.loc[(tLower <= self.df.Minute) & (self.df.Minute <= tUpper)].copy(deep=True)

                    ft_y = fft(timeSeries.values)
                    ft_y = pd.Series(2/N * np.abs(ft_y[:N//2])**2) # Multiplying by 2 to deal with the fact that we chopped out -ive freqs

                    title = fileName[:len(fileName) - 4] + plotTitle + f"_{tLower}-{tUpper}MIN_" + entry

                    logical = ft_y > min(ft_y) # Removing erroneous minimum value which is << the second smallest
                    ft_x = ft_x[logical]
                    ft_y = ft_y[logical]

                    plt.loglog(ft_x, ft_y, plotType, color='b')
                    plt.xlabel('Frequency Domain', fontsize=5)
                    plt.ylabel(f'Spectral Density of Clean {entry} Data', fontsize=5)
                    plt.title(title, fontsize=5)

                    self._gradient_line_aux(ft_y, ft_x, gradient)

                    xVal = ft_x[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y)))]
                    yVal = ft_y.loc[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y))) - 1] # -1 to stop extra point being added in
                    m, _, pearson_r, _, _ = linregress(np.log10(xVal), np.log10(yVal))
                    if supervised:
                        plt.show()
                    elif np.abs(pearson_r) < pearson_cutoff and np.abs(gradient - m) > gradient_cutoff:
                        print(f"Rejected {fileName}: Pearson R = {pearson_r}, m = {m}")
                        plt.close()
                        return True
                    else:
                        plt.savefig(os.path.join(saveLoc, f"{title}.png"))
                        plt.close()

                    return False
        
        else:
            N = len(self.df)
            ft_x = fftfreq(N, sampleSpacing)[:N//2] # The function is symmetric so we are only interested in +ive frequency values
            ft_x = ft_x/(2*np.pi)

            ft_y = fft(self.df[entry].values)
            ft_y = pd.Series(2/N * np.abs(ft_y[:N//2])**2) # Multiplying by 2 to deal with the fact that we chopped out -ive freqs

            title = fileName[:len(fileName) - 5] + plotTitle + entry
        
            logical = ft_y > min(ft_y) # Removing erroneous minimum value which is << the second smallest
            ft_x = ft_x[logical]
            ft_y = ft_y[logical]

            plt.loglog(ft_x, ft_y, plotType, color='b')
            plt.xlabel('Frequency Domain')
            plt.ylabel(f'Spectral Density of Clean {entry} Data')
            plt.title(title)

            self._gradient_line_aux(ft_y, ft_x, gradient)

            xVal = ft_x[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y)))]
            yVal = ft_y.loc[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y))) - 1] # -1 to stop extra point being added in
            m, _, pearson_r, _, _ = linregress(np.log10(xVal), np.log10(yVal))
            if supervised:
                plt.show()
            elif np.abs(pearson_r) < pearson_cutoff and np.abs(gradient - m) > gradient_cutoff:
                print(f"Rejected {fileName}: Pearson R = {pearson_r}, m = {m}")
                plt.close()
                return True
            else:
                plt.savefig(os.path.join(saveLoc, f"{title}.png"))
                plt.close()

            return False

    def _window_width_aux(self, N: int, timeSeries: pd.DataFrame, windowWidth: int, entry: str, tLower: float, tUpper: float):
            # Preallocating an array of FFTs over windowWidth long windows to average over
            windows = range((tUpper - tLower)//windowWidth)
            ft_y_arr = pd.DataFrame(np.zeros([len(windows), N//2]))

            # Going over each windowWidth snapshot and FFTing
            for i in windows:
                y = timeSeries.loc[(tLower + i*windowWidth <= self.df.Minute) & (self.df.Minute <= tUpper - (len(windows) - i - 1)*windowWidth), entry]
                y_bar = y.mean()
                y_turb = y - y_bar
                y_turb.dropna(inplace=True)

                try:
                    ft_yTemp = fft(y_turb.values)
                    ft_yTemp = 2/N * np.abs(ft_yTemp[:N//2])**2 # Multiplying by 2 to deal with the fact that we chopped out -ive freqs
                    ft_y_arr.loc[i] =  ft_yTemp
                except ValueError: # If it's an empty array and errors out, we reject it
                    return None

            # Averaging over all time snapshots
            return ft_y_arr.mean(axis = 0)
    
    def _gradient_line_aux(self, ft_y: pd.Series, ft_x: pd.Series, gradient: int):
            # Using y = ax^k for loglog plotting
            yVal = ft_y.loc[int(np.floor(0.05*len(ft_y)))] # Making the line cut through the approximate start of the tail (the tail corresponds to the tail of the FT, which begins at around 1% of the domain length due to the steep slope near the start of the FT)
            xVal = ft_x[int(np.floor(0.05*len(ft_y)))]
            c = np.log10(yVal/(xVal**gradient))

            # Restricting the line to 1-50% of the domain to prevent the line from diverging too far away from the main plot and affecting the scale
            ft_x = ft_x[int(np.floor(0.01*len(ft_y))):int(np.floor(0.5*len(ft_y)))]
            line = (10**c)*ft_x**gradient # Guarding against div0 errors

            plt.loglog(ft_x, line, color='r')
            plt.legend(['FFT', f'm = {round(gradient, 2)}'])

    def plot_hist(self, entry: str, fileName: str, diffCutOff: float, supervised=False, saveLoc=None, plotTitle="_HIST_", bins=50):
        """
        Plots a histogram of entry with the specified amount of bins. Does not do so and returns True if the two neighboruing count values are more
        than diffCutOff times larger than each other. Refer to plot_comparison for other parameters.
        """
        if self.reject_hist_outliers(entry, diffCutOff):
            print(f"Rejected {fileName}: Histogram has a spike")
            return True
        else:
            sns.histplot(data=self.df, x=entry, bins=bins)

            title = fileName[:len(fileName) - 4] + plotTitle + entry

            plt.xlabel(entry)
            plt.ylabel('Frequency (no. occurences)')
            plt.title(title)

            if supervised:
                plt.show()
            else:
                plt.savefig(os.path.join(saveLoc, f"{title}.png"), dpi=500)
                plt.close()

            return False
        
    def reject_file_on_changing_mean(self, entry: str, margain:float, sec_stepsize: int, n_most=100) -> bool:
        s = self.df[entry]
        window_width = sec_stepsize // (self.df.GlobalSecs[1] - self.df.GlobalSecs[0]) # Amount of indicies to consider = wanted_stepsize/data_stepsize
        windows = s.rolling(window=window_width, step=window_width)

        means = windows.mean()
        n_largest = means.nlargest(n=n_most)
        n_smallest = means.nsmallest(n=n_most)
        return (n_largest.mean() - n_smallest.mean()) > margain

    def reject_file_on_range(self, entry: str, margain: float, std_devs=2, sec_stepsize=3600) -> bool:
        '''
        Checks if the avg range of the data (i.e. 2*standard devs) is larger margain.

        :param entry: (str) The parameter key.
        :param margain: (float) Rejects file if std_devs*std > margain (i.e. if N standard deviations > a given margain, the range is too wide).
        :param std_devs: (float) Amount of stds away from mean to consider as the range.
        :param sec_stepsize: (float) The amount of seconds to look at at a time. The whole dataset by default.
        :return: (bool) True if rejected, False if not.
        '''
        s = self.df[entry]
        window_width = int(sec_stepsize // (self.df.GlobalSecs[1] - self.df.GlobalSecs[0])) # Amount of indicies to consider = wanted_stepsize/data_stepsize
        windows = s.rolling(window=window_width, step=window_width)

        stds = windows.std()
        return (std_devs*stds > margain).any()
    
    def reject_file_on_gradient(self, entry: str, margain: float, sec_stepsize=None) -> bool:
        '''
        Checks if data gradients are too extreme and removes it if so.

        :param entry: (str) The parameter key.
        :param margain: (float) Rejects file if avg_deviation > margain.
        :param sec_stepsize: (float) The amount of seconds to look at at a time.
        :return: (bool) True if rejected, False if not.
        '''
        lwr = 0
        if sec_stepsize is None:
            sec_stepsize = self.df.GlobalSecs.max()
            upr = sec_stepsize
        else:
            upr = sec_stepsize

        while upr <= self.df.GlobalSecs.max():            
            slice = self.df[(self.df.GlobalSecs >= lwr) & (self.df.GlobalSecs <= upr)]
            # Finding the derivative in unit/s
            dt = slice.loc[pd.notna(slice[entry]), 'GlobalSecs'].diff()
            dy = slice.loc[pd.notna(slice[entry]), entry].diff()
            slopes = dy.div(dt) # dy/dt

            deviation = np.abs(slopes - slopes.mean())

            if deviation.mean() > margain:
                return True
            
            upr += sec_stepsize
            lwr += sec_stepsize

        return False
    
    def std_cutoff(self, entry: str, stdMargain: float, sec_stepsize: float) -> pd.Series:
        """
        Returns logicals set to True for data of type entry that lies beyond +-stdMargain standard deviations from the mean of the dataset over sec_stepsize seconds (Over the entire dataset if None).
        """
        s = self.df[entry]

        sec_0, sec_1 = self.df.GlobalSecs.nsmallest(n=2)
        window_width = int(sec_stepsize // (sec_1 - sec_0)) # Amount of indicies to consider = wanted_stepsize/data_stepsize
        windows = s.rolling(window=window_width, step=window_width)

        logical = pd.Series(len(s)*[False])
        for window in windows:
            std = window.std()
            mean = window.mean()

            if pd.isna(std):
                logical[window.index] = pd.Series(len(window)*[False])
            else:
                logical[window.index] = np.abs(s[window.index] - mean) > stdMargain*std

        return logical

    def gradient_cutoff(self, entry: str, diffStdMargain: float) -> pd.Series:
        """
        Returns logicals set to True for type entry data whose gradients lie beyond +-diffStdMargain standard deviations from the mean slope (unit/s).
        """
        # Finding the derivative in unit/s
        dt = self.df.loc[pd.notna(self.df[entry]), 'GlobalSecs'].diff()
        dy = self.df.loc[pd.notna(self.df[entry]), entry].diff()
        slopes = dy.div(dt) # dy/dt

        cutOff = diffStdMargain*slopes.std()
        mean = slopes.mean()

        # Finding slopes too steep and returning associated datapoints
        slopeIdx = slopes.index[np.abs(mean - slopes) > cutOff].to_series()
        return self.df.index.isin(slopeIdx)

    def reject_hist_outliers(self, entry: str, diffCutoff: float, counts=20) -> bool:
        """
        Checks for values with unusually high frequencies and returns true if the any bin is diffCutoff times larger than the next largest. Only
        checks the first counts counts
        """
        freqDf = self.df.groupby([entry])[entry].count().reset_index(name='Count').sort_values(['Count'], ascending=False)
        idxSer = freqDf.index.to_series().reset_index()
        # Grabbing first counts counts since the latter are small anyway
        idxSer.drop(labels=range(counts+1, len(idxSer)), inplace=True)
        idxSer.drop(0, axis="columns", inplace=True)
        idxSer.rename(columns={"index":"countIdx"}, inplace=True)

        # Checking each pair and breaking out if it's too steep. Otherwise return false
        for i in range(len(idxSer) - 1):
            idx = idxSer.countIdx[i]
            idx2 = idxSer.countIdx[i + 1]
            if freqDf.Count[idx]/freqDf.Count[idx2] >= diffCutoff:
                return True
        return False

    def prune_and(self, entry: str, logicals: pd.Series, naive_nans=False) -> None:
        """
        Cuts out datapoints of type entry which fit into the intersection of all conditions in the tuple logicals and replaces them
        with linear interpolations.
        """
        logical = logicals[0]

        # Combining logicals into single condition
        if len(logicals) > 1:
            for log in logicals:
                logical = logical & log

        self.df = self.df[~logical]
        self.df.reset_index(drop=True, inplace=True)

    def prune_or(self, entry: str, logicals: pd.Series, naive_nans=False) -> None:
        """
        Cuts out datapoints of type entry which fit into the union of all conditions in the tuple logicals and replaces them
        with linear interpolations.
        """
        logical = logicals[0]

        # Combining logicals into single condition
        if len(logicals) > 1:
            for log in logicals:
                logical = logical | log

        self.df = self.df[~logical]
        self.df.reset_index(drop=True, inplace=True)
    
    def mru_correct(self) -> None:
        """
        Updates anemometer velocities to be motion corrected with the MRU. Currently only works for anem1 (14.8m), however it adjusts it
        so that u+ is North and v+ is East in both.
        """
        w1 = "Anemometer #1 W Velocity (ms-1)"
        w2 = "Anemometer #2 W Velocity (ms-1)"
        u1 = "Anemometer #1 U Velocity (ms-1)"
        u2 = "Anemometer #2 U Velocity (ms-1)"
        v1 = "Anemometer #1 V Velocity (ms-1)"
        v2 = "Anemometer #2 V Velocity (ms-1)"
        comp1 = "Compass #1 (deg)"
        comp2 = "Compass #2 (deg)"
        mru_pitch = 'MRU Pitch Angle (deg)'
        mru_roll = 'MRU Roll Angle (deg)'
        mru_p = 'MRU P Axis Velocity'
        mru_r = 'MRU R Axis Velocity'
        mru_y = 'MRU Y Axis Velocity'

        # Corrrecting anem bug
        log1_1 = self.df[w1] > 0
        log1_2 = self.df[w1] < 0
        self.df.loc[log1_1, w1] = self.df.loc[log1_1, w1]*1.166
        self.df.loc[log1_2, w1] = self.df.loc[log1_2, w1]*1.289
        log2_1 = self.df[w2] > 0
        log2_2 = self.df[w2] < 0
        self.df.loc[log2_1, w2] = self.df.loc[log2_1, w2]*1.166
        self.df.loc[log2_2, w2] = self.df.loc[log2_2, w2]*1.289

        # Converting compass to east-north components
        comp1_copy = self.df[comp1].copy(deep=True)
        log1 = self.df[comp1] > 180
        comp1_copy[log1] = comp1_copy[log1] - 360
        comp2_copy = self.df[comp2].copy(deep=True)
        log2 = self.df[comp2] > 180
        comp2_copy[log2] = comp2_copy[log2] - 360

        # Correcting for wire motion (only in anem 1 since anem 2 doesn't have an MRU. -v is east, u is north. Anem 1 is higher up - 14.8m)
        #NOTE: In future, put this before prunes
        v1_temp = -self.df[v1] - (-self.df[mru_r])
        u1_temp = self.df[u1] - self.df[mru_p]
        w1_temp = self.df[w1] #TODO: Why no mru_y?
        self.df[v2] = -self.df[v2] # Flipping so that v points east in anem 2 as well
        
        comp1_copy = np.deg2rad(comp1_copy - 180)
        # \ indicates we're continuing on the next line
        self.df[u1] = u1_temp*np.cos(comp1_copy)*np.cos(np.deg2rad(self.df[mru_pitch])) \
                    + v1_temp*(-np.sin(comp1_copy)*np.cos(np.deg2rad(self.df[mru_roll])) + np.cos(comp1_copy)*np.sin(np.deg2rad(self.df[mru_pitch]))*np.sin(np.deg2rad(self.df[mru_roll]))) \
                    + w1_temp*(np.sin(comp1_copy)*np.sin(np.deg2rad(self.df[mru_roll])) + np.cos(comp1_copy)*np.sin(np.deg2rad(self.df[mru_pitch]))*np.cos(np.deg2rad(self.df[mru_roll])))
        
        self.df[v1] = u1_temp*np.sin(comp1_copy)*np.cos(np.deg2rad(self.df[mru_pitch])) \
                    + v1_temp*(np.cos(comp1_copy)*np.cos(np.deg2rad(self.df[mru_roll])) + np.sin(comp1_copy)*np.sin(np.deg2rad(self.df[mru_pitch]))*np.sin(np.deg2rad(self.df[mru_roll]))) \
                    + w1_temp*(np.sin(np.deg2rad(self.df[mru_pitch]))*np.cos(np.deg2rad(self.df[mru_roll]))*np.sin(comp1_copy) - np.sin(np.deg2rad(self.df[mru_roll]))*np.cos(comp1_copy))

        self.df[w1] = -u1_temp*np.sin(np.deg2rad(self.df[mru_pitch])) \
                    + v1_temp*np.cos(np.deg2rad(self.df[mru_pitch]))*np.sin(np.deg2rad(self.df[mru_roll])) \
                    + w1_temp*np.cos(np.deg2rad(self.df[mru_pitch]))*np.cos(np.deg2rad(self.df[mru_roll]))

        # Sign flipping to correct for bug
        # TODO NOT DONE FOR ANEM 2
        self.df[u1] = -self.df[u1]
        self.df[v1] = -self.df[v1]

    def remove_nans(self, entry: str, df: pd.DataFrame, naive=False) -> None:
        """
        Cuts out NaNs in entry and removes them with linear interpolations. If naive, it simply takes means assuming neighbours are equally
        spaced and we no double-missing points.
        """
        nanIdx = df.loc[pd.isna(df[entry])].index.to_series()
        nanIdx.apply(lambda x: self._remove_nans_aux(entry, df, x, naive=naive))
        df[entry].fillna(method='bfill')

    def _remove_nans_aux(self, entry: str, df: pd.DataFrame, nanIdx: int, naive=False) -> None:
        # If neighbouring points are NaN, recursively find the nearest points which aren't
        #xLower, xUpper = self._interp_aux(entry, nanIdx - 1, nanIdx + 1)
        xLower = nanIdx
        xUpper = nanIdx

        while pd.isna(df.loc[xLower, entry]):
            if xLower >= 0:
                xLower -= 1
            if xLower < 0:
                break
        while pd.isna(df.loc[xUpper, entry]):
            if xUpper <= len(df) - 1:
                xUpper += 1
            if xUpper > len(df) - 1:
                break

        if naive:
            # Just taking means of neighbours without caring about distance. If we need to extrapolate, just stick it to the edgepoint
            if xLower >= 0 and xUpper <= len(df) - 1:
                # Catching edge cases where angles reset from 180 or 360 back to 0
                if (df.loc[xLower, entry] < 10 and df.loc[xUpper, entry] > 10) or (df.loc[xLower, entry] > 10 and df.loc[xUpper, entry] < 10):
                    df.loc[nanIdx, entry] = df.loc[xLower, entry]
                else:
                    df.loc[nanIdx, entry] = np.mean([df.loc[xLower, entry], df.loc[xUpper, entry]])

            elif xLower < 0 and xUpper < len(df) - 1:
                df.loc[nanIdx, entry] = df.loc[xUpper, entry]

            elif xLower > 0 and xUpper > len(df) - 1:
                df.loc[nanIdx, entry] = df.loc[xLower, entry]

            else:
                raise ValueError("This dataset is scuffed. Every single point was flagged as bad.")

        else:
            # Finding neighbouring x and y values to interpolate between
            if xLower >= 0 and xUpper <= len(df) - 1:
                xNeighbours = df.GlobalSecs[[xLower, xUpper]].values
                yNeighbours = df.loc[[xLower, xUpper], entry].values

                df.loc[nanIdx, entry] = np.interp(df.GlobalSecs[nanIdx], xNeighbours, yNeighbours) # Linearly interpolating. If we need to extrapolate (i.e. endpoints), we just say that the value = the neighbour

            elif xLower < 0 and xUpper < len(df) - 1:
                xNeighbours = [df.GlobalSecs[xUpper], df.GlobalSecs[xUpper + 1]] # Forcing the interpolator to extrapolate when we have an edgepoint
                yNeighbours = [df.loc[xUpper, entry], df.loc[xUpper + 1, entry]]

                df.loc[nanIdx, entry] = np.interp(df.GlobalSecs[nanIdx], xNeighbours, yNeighbours, left=yNeighbours[0], right=yNeighbours[0])

            elif xLower > 0 and xUpper > len(df) - 1:
                xNeighbours = [df.GlobalSecs[xLower - 1], df.GlobalSecs[xLower]]
                yNeighbours = [df.loc[xLower - 1, entry], df.loc[xLower, entry]]

                df.loc[nanIdx, entry] = np.interp(df.GlobalSecs[nanIdx], xNeighbours, yNeighbours, left=yNeighbours[1], right=yNeighbours[1])

            else:
                raise ValueError("This dataset is scuffed. Every single point was flagged as bad.")

    # def _interp_aux(self, entry, left, right) -> int:
    #     """
    #     Recursively searching for the nearest non-NaN value to interpolate to. left is the index left of the value being interpolated. Vice versa with
    #     right.
    #     """
    #     if right < len(df) - 1 and pd.isna(df.loc[right, entry]): # Need to check if right isn't already an edgepoint
    #         return self._interp_aux(entry, left, right + 1)

    #     elif left > 0 and pd.isna(df.loc[left, entry]):
    #         return self._interp_aux(entry, left - 1, right)
        
    #     else:
    #         return (left, right)