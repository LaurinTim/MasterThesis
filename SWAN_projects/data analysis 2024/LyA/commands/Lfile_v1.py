#!/usr/bin/env python
# coding: utf-8

# In[385]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')
sys.path.append('/eos/user/l/lkoller/data-analysis-software')
sys.path.append('/eos/user/l/lkoller/SWAN_projects/commands/data_loader')

from pathlib import Path
import pandas as pd
import numpy as np
from readTrc_4CH import Trc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import os
from datetime import datetime
import time
import scipy.signal as sci


# In[469]:


#class for 1 trc file with data from the lyman alpha detectors
class Lfile():
    '''
    
    Class for a single trc file which gives info like the peak locations and heights and whether the mw was off/on for the measurement
    
    Parameters
    ------------
    path of the file
    
    '''
    
    def __init__(self, filepath):
        self.filepath = filepath
        
    def __str__(self):
        return f'{self.filepath}'
    
    #read trc file
    def read_trc(self):
        '''
        get the data from the trc file
        
        '''
        read = Trc() #define element of class trc
        return read.open(Path(self.filepath))
    
    
    #check if a file exists at filepath
    def check_filepath(self):
        '''
        check if the file exists
        
        '''
        return Path(self.filepath).exists()
        
        
    def run_number(self):
        '''
        
        Check what run the current file is a part of
        
        Returns
        ------------
        the run number of the file
        
        '''
        df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
        run = list(df[df['LyA'] == self.filepath]['run number'])[0]
        
        return run
    
    
    #function to determine whether mw was on (1) or off (0)
    def mw_on(self):
        '''
        
        look whether the mw was turned on or off for the file
        
        Returns
        ------------
        1 if mw was on
        0 if mw was off
        
        '''
        d = str(datetime.fromtimestamp(int(self.filepath[-18:-8])).date())
        if d in ['2024-04-23', '2024-04-25']:
            date = d[2:4] + '_' + d[5:7] + '_' + d[8:]
            pdf = pd.DataFrame(np.load('/eos/user/l/lkoller/GBAR/data24/datasummary24/' + date + '/LyA_data/peaks_' + date + '.npy', allow_pickle = True), 
                  columns = ['LyA', 'microwave', 'time ch1', 'height ch1', 'time ch2', 'height ch2', 'time ch3', 'height ch3', 'time ch4', 'height ch4'])
            if str(pdf[pdf['LyA'] == self.filepath]['microwave']) == 'on':
                return 1
            if str(pdf[pdf['LyA'] == self.filepath]['microwave']) == 'off':
                return 0
            else:
                return print('error')
        
        df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
        mw_power = list(df[df['LyA'] == self.filepath]['MW power'])[0]
        
        if (mw_power > 0.0001): #take the third channel because it is most visible there
            return 1
        else:
            return 0
    
    
    #get events from one file, return list with 4 lists where we can see where in the data of the 4 seperate channels there are peaks and a list with the peak heights
    #mw will return 0 (if the mw is turned off) or 1 (if the mw is turned on)
    def get_events_from_file(self, prom = 0.001, hgt = 0.005, dist = 30, l = 30, trange = None):
        '''
        get information about the events recorded by the 4 mcps in the lya setup for the file
        
        Parameters
        ------------
        prom = 0.001 is a parameter used to determine the peaks
        hgt = 0.003 is the minimum height for a peak to be recorded
        dist = 30 the distance two peaks have to be apart from eachother
        l = 30 how many datapoints we go back to determine the std, with which we determine whether something is a peak or not
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        
        Returns
        ------------
        events = list of times at which events were registered for each channel
        p_height = list of heights of the events for each channel
        self.mw_on() is 1 if the mw is on and 0 if it is off for the file
        
        '''
        if trange == None: cut_low, cut_high = 0,10002
        else: cut_low, cut_high = trange[0], trange[1]
        
        #check if file exists
        if (self.check_filepath == False):
            return print('The file at ' + self + ' does not exist.')
        
        data = self.read_trc() #read the trc file  
        time = data[0][0] #define time which we use for all 4 detectors (off by around 0.5 ns for 2 of the 4 detectors, but this should not matter)
        volt = [-data[1][0], -data[1][1], -data[1][2], -data[1][3]] #voltage data for the four channels
        
        events = [0,0,0,0] #list to put the lists where the events are into   
        p_height = [0,0,0,0] #list for the peak heights of the 4 detectors
        mw = 0 #return mw = 1 if mw was on, otherwise mw = 0
        
        #go through the data of the four detectors
        for i in range (4):
            #get the average for the current channel and subtract it from the voltage
            #avg = np.average(volt[i][8000:10000])
            #volt[i] = [val - avg for val in volt[i]]
            
            peaks = sci.find_peaks(volt[i], prominence = prom, height = hgt, distance = dist) #these parameters define what we exactly determine to be a peak, the current ones could be bad
            events[i] = peaks[0] #set the ith element of the list 'events' to the locations of the peaks
            events[i] = [val for val in events[i] if val > cut_low and val < cut_high]
            #p_height[i] = -peaks[1]['peak_heights'] #put peak heights into p_heights, put the minus because in the data for sci.find_peaks we put a minus
            
            for m in range(len(events[i])):
                if (l >= 10):
                    tl = volt[i][(events[i][m]-l) : (events[i][m]-10)] #list with voltage for the datapoints from l before the mth peak of the ith channel until 10 before the peak
                    
                    #if tl has 0 elements (so the peak would be in the first 40 datapoints) just set it to [0, 1] so the peak gets ignored
                    if (len(tl) == 0): 
                        tl = [0, 1]

                    mean = sum(tl) / len(tl) #mean of the elements in tl
                    variance = sum([((x - mean) ** 2) for x in tl]) / len(tl) #variance of the elements in tl
                    std = variance ** 0.5 #standard deviation of the elements in tl

                    #look at average value of 5 elements before peak, if it is too small discard peak
                    av = sum([val for val in volt[i][(events[i][m]-5):events[i][m]]])/5
                    
                    #we want to disregard some elements depending on their peak height and the standard deviation of tl and the max element of tl
                    if (((std/volt[i][events[i][m]] > 0.15 and volt[i][events[i][m]] < 0.06) and 
                        (max(tl) >= volt[i][events[i][m]]/2 or min(volt[i][events[i][m]:events[i][m]+15]) <= -volt[i][events[i][m]]/2 or max(volt[i][events[i][m]+10:events[i][m]+25]) >= volt[i][events[i][m]]/1.5)) 
                        or av < 0.001):
                        events[i][m] = -1
                        
                    if volt[i][events[i][m]] < 0.01 and volt[i][events[i][m] - 5] > 0.001:
                        events[i][m] = -1
                       
                else:
                    #look at average value of 5 elements before peak, if it is too small discard peak
                    av = sum([(val) for val in volt[i][(events[i][m]-5):events[i][m]]])/5

                    #we want to disregard some elements depending on their peak height and the standard deviation of tl
                    #if ((std/volt[i][events[i][m]]) > 0.15 and volt[i][events[i][m]] < 0.06 or av < 0.001):
                    if (av < 0.001):
                        events[i][m] = -1
                if (events[i][m] <= l):
                    events[i][m] = -1

            events[i] = [val for val in events[i] if val != -1] #only keep events that are not at position -1
            p_height[i] = [volt[i][val] for val in events[i]] #put the peak heights into p_height
        
        d = str(datetime.fromtimestamp(int(self.filepath[-18:-8])).date())
        if d in ['2024-04-23', '2024-04-25']:
            return events, p_height, self.mw_on(), [0] * len(events)
        
        return events, p_height, self.mw_on(), self.run_number()
    
    
    #plot the data from the ith channel (channel 5 for all of them added, this is the default)
    #keep in mind that the peaks for channel 5 are where the peaks for the single channels are, so it might look weird
    #use channel 6 to plot all 4 channels in 1 plot
    def plot_voltage(self,p = 0.001, h = 0.005, d = 30, back = 30, trange = None, comb = False):
        '''
        plot the voltage recorded for the channels
        
        Parameters
        ------------
        p = 0.001 is a parameter used to determine the peaks
        h = 0.005 is the minimum height for a peak to be recorded
        d = 30 the distance two peaks have to be apart from eachother
        back = 30 how many datapoints we go back to determine the std, with which we determine whether something is a peak or not
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        comb: default False, if True then only a plot with the combined voltages and peaks from the 4 channels is output

        '''
        if trange == None: cut_low, cut_high = 0,10002
        else: cut_low, cut_high = trange[0], trange[1]
        
        data = self.read_trc() #get the data for the file
        t, h, m, r = self.get_events_from_file(prom = p, hgt = h, dist = d, l = back, trange = trange) #get the time and heights for the peaks

        time = data[0][0] #use this for the time
        volt = [data[1][0], data[1][1], data[1][2], data[1][3]] #list with the voltages for the different channels
        time_peaks = [list(t[0]), list(t[1]), list(t[2]), list(t[3])] #list with the times of the peaks for the different channels
        height_peaks = [list(h[0]), list(h[1]), list(h[2]), list(h[3])]
        
        #get the average for the current channel and subtract it from the voltage
        #for i in range (4):
        #    av = np.average(volt[i][8000:10000])
        #    volt[i] = [val - av for val in volt[i]]
        
        if comb:
            fig = plt.figure(figsize = (15,8))
            
            for i in range(4):
                chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
                plt.scatter(time[cut_low:cut_high], volt[i][cut_low:cut_high], label = 'Voltage channel ' + str(i))
                plt.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 100)
                
            plt.legend(loc = 'best')
            plt.xlabel(xlabel = 'Time [µs]')
            plt.ylabel(ylabel = 'Voltage [V]')
            plt.title(label = 'Voltages and peaks of the four channels')
            return
        
        fig = plt.figure(layout = 'tight', figsize = (15,10))
        gs = GridSpec(2, 2, figure = fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        
        for i, ax in enumerate(fig.axes):
            chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
            ax.scatter(time[cut_low:cut_high], volt[i][cut_low:cut_high], s = 15)
            ax.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 30, linewidth = 3)
            #ax.grid(True, alpha = 1)
            ax.set_ylabel(ylabel = 'Voltage [V]')
            ax.set_xlabel(xlabel = 'Time [µs]')
            ax.set_title(label = 'Voltage and recorded peak in channel %i' % (i + 1))
        
        return
