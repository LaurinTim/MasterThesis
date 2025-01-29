#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data loader')
sys.path.append('/eos/home-i00/l/lkoller/data-analysis-software')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from readTrc_4CH import Trc
import os
from datetime import datetime
from tqdm import tqdm
from gbarDataLoader24 import loadShortSummary


# In[3]:


class wf:
    '''
    
    A class to get information from the waveform files
    
    Parameters
    ------------
    date = yy_mm_dd day from which we evaluate the waveform files
    
    '''
    
    
    def __init__(self, date):
        self.date = date
    
    
    def __str__(self):
        return f'{self.date}'
    
    
    def read(filepath):
        '''
        get the data from the trc file
        
        Parameters
        ------------
        filepath = path to a file

        '''
        read = Trc() #define element of class trc
        return read.open(filepath)
    
    
    def filelist(self):
        '''
        
        Get a list with the filepaths of the waveform files
        
        Returns
        ------------
        flist = list with the path to all the waveform files from the current day
        
        '''
        wfpath = Path('/eos') / 'experiment' / 'gbar' / 'pgunpc' / 'data' / str(self.date)
        flist = [str(wfpath / val) for val in os.listdir(wfpath) if val.startswith('WF1234') and val.endswith('.trc')]
        return flist
    
    
    def measurement_times(self):
        '''
        
        get the times at which the files were generated
        
        Returns
        ------------
        times[0] = timestamps from the files
        times[1] = datetimes from the files
        
        '''
        times = [[],[]]
        
        files = self.filelist()
        
        for i in files:
            temp = i[-18:-8]
            times[0] += [temp]
            times[1] += [datetime.fromtimestamp(int(temp))]
        
        return times
    
    
    def windows(self):
        '''
        
        get a list with the times at which the beam arrives at MCP5 (channel 3 in the waveform file) for the current day
        
        Returns
        ------------
        df = dataframe with the arrival windows, file names and run numbers for all the files. The columns are:
            Time: timestamp from when the file was generated
            datetime: datetime from Time
            Waveform: path to the file for the current row
            run: run number of the file
            NE50_I: number of particles in millions in the NE50 line (so how many particles we get from elena)
            MCP5_volt_sum: sum of the voltages of the MCP5 in the timeframe of the window
            beam_start = time when the beam starts arriving at MCP5
            beam_stop = time when the beam stops arriving at MCP5
                These times are given as the difference in time from the first time element in the waveform file. To get to the time in the LyA file, for example, one needs to subtract 5e-6 - 3e-7
            
        '''
        files = self.filelist()
        times = self.measurement_times()
        timestamps = times[0]
        datetimes = times[1]
        start = [] #list with the start times of the window
        stop = [] #list with the stop times of the window
        runs = [] #run numbers
        ne50 = [] #NE50 line intensity
        voltl = [] #list with the sum of the voltages in the window
        
        summary = loadShortSummary(self.date)
        
        for i in tqdm(files, total = len(files)):
            runs += list(summary[summary['Waveform_12bit'] == i]['run'])
            ne50 += list(summary[summary['Waveform_12bit'] == i]['NE50_I'])
            try:
                data = wf.read(i)
            except:
                voltl += [0]
                start += [0]
                stop += [0] 
            else:
                if ne50[-1] > 1:
                    st = data[0][2][0]
                    temp_window, _, vsum = self.fwindow(i)
                    temp_window = [val - st for val in temp_window]
                    voltl += [vsum]
                    start += [temp_window[0]]
                    stop += [temp_window[1]]
                else:
                    voltl += [0]
                    start += [0]
                    stop += [0]
                                
        df = pd.DataFrame([timestamps, datetimes, files, runs, ne50, voltl, start, stop], index = ['Time', 'datetime', 'MCP5_Waveform', 'run', 'NE50_I', 'MCP5_volt_sum', 'beam_start', 'beam_stop']).transpose()
        
        df = df.fillna('None')
        df = df.replace('NaN', 'None')
        
        return df
    
    
    def fwindow(self, file):
        '''
        
        get the window in which the beam arrives at MCP5 (channel 3 in the waveform file) from one file
        
        Parameters
        ------------
        file = path to the file which we are looking at
        
        Returns
        ------------
        List with the time of the beam start and stop
        List with the timestep of the beam start and stop
        vsum: sum of all the voltages in the window
        
        '''
        data = wf.read(file)
        x = np.arange(0, len(data[0][2]))#[4000:9000]
        y = data[1][2]#[4000:9000]
        av = sum(y[9000:])/3502
        y = [val - av for val in y]
        
        n = 50
        xaxis = x[::n]
        yaxis = np.zeros(len(xaxis))
        
        for i in range(len(xaxis)):
            yaxis[i] = np.average(y[i * n: (i+1) * n])
        
        ymax = np.argmax(yaxis[60:-1]) + 60
        xlow = ymax
        xhigh = ymax
        
        if self.date in ['24_07_30', '24_07_31']:
            thresh = 0.0005 + 2*np.std(yaxis[-70:-1])
        else:
            thresh = 0.0003 + 2*np.std(yaxis[-70:-1])
                
        while xlow >= 5 and ((yaxis[xlow] > thresh or yaxis[xlow-1] > thresh) and yaxis[xlow+1] > thresh or (yaxis[xlow-1] < yaxis[xlow] and yaxis[xlow] > 0.0001)):
            xlow -= 1
        
        while xhigh <= (len(xaxis) - 6) and ((yaxis[xhigh] > thresh or yaxis[xhigh+1] > thresh) and yaxis[xhigh-1] > thresh or (yaxis[xhigh+1] < yaxis[xhigh] and yaxis[xhigh] > 0.0001)):
            xhigh += 1
        
        xlow -= 4
        xhigh += 4
        if xhigh > 250: xhigh = 250
        if xlow < 0: xlow = 0
            
        vsum = sum(y[xaxis[xlow]:xaxis[xhigh]])
        
        return [data[0][2][xaxis[xlow]], data[0][2][xaxis[xhigh]]], [xaxis[xlow], xaxis[xhigh]], vsum
    
    
    def wf_data(self):
        '''
        
        write a text file with the output from windows for the current date
        
        '''
        LYAFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date / 'LyA_data'
        
        if((LYAFILE / ('wf_' + self.date + '.txt')).is_file()):
            print('The waveform datafile for ' + self.date + ' already exists.')
            return pd.read_csv(str(LYAFILE) + '/wf_' + self.date + '.txt', sep = '\t')
        
        df = self.windows()
        
        df.to_csv(str(LYAFILE) + '/wf_' + self.date + '.txt', sep = '\t', index = False)
        
        return df
