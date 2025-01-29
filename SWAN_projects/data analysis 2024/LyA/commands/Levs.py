#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
import os
from Lfile import Lfile
from Ldate import Ldate
from gbarDataLoader24 import loadShortSummary
from readTrc_4CH import Trc
from LyAdata24 import read_df
from tqdm import tqdm
from datetime import datetime
import math
from ast import literal_eval
pd.set_option("display.max_rows",90)
pd.set_option("display.max_columns",None)


# In[2]:


tof_start = - 8.6e-7 #time we have to subtract from the time in the MCP5 waveform where the beam starts passing through the LyA detectors
tof_stop = - 7.0e-7 #time we have to subtract from the time in the MCP5 waveform where the beam stops passing through the LyA detectors

class Levs:
    '''
    
    Analyze the data for the peaks data
    
    Parameters
    ------------
    The input should be a dataframe containing the time and height of peaks as well as their channel and microwave status and corresponding LyA file.
    Use the files generated with Ldate for the correct information
    
    
    '''
    
    #define instance method
    def __init__(self, evs):
        self.evs = evs
    
    
    #convert the timestep to the actual time (from 0 - 10002 to -5e-6 - 5e-6)
    def conv(i):
        ts = -4.999745000028291e-06
        tf = 5.001254717124112e-06
        tw = tf - ts
        l = 10001
        dt = tw/l
        return ts + dt * i
    
    #convert the actual time to the timestamp(from -5e-6 - 5e-6 to 0 - 10002)
    def conv2(i):
        ts = -4.999745000028291e-06
        tf = 5.001254717124112e-06
        tw = tf - ts
        l = 10001
        dt = tw/l
        return (i - ts) / dt
    
    
    def norm_method(self, method, vrange = None, trange = None, tof = False):
        '''
        
        get the value to adjust the microwave on bar heights for different normalization methods
        
        Parameters
        ------------
        method: the method we want to use for the normalization
            max elements: we look at how many peaks mw on/off have at the maximum peak height and multiply the amount of peaks in the bins for mw on by their ration (off/on)
            filenumber: count the amount of files for mw on/off and multiply the amount of peaks in the bins for mw on by their ration (off/on)
            elena: count for each file the amount of particles in the NE50 line and multiply the amount of peaks in the bins for mw on by their ration (off/on)
        vrange = the voltage range in which we want to look at the peak
            input as a list with vrange[0] < vrange[1], default is [0.005, 0.1453334428369999 ]
            the measured voltage only goes up to 0.1453334428369999, so all peaks that would be higher are just exactly at this voltage
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
            
        Returns
        ------------
        adj: a float that should be multiplied with each bin height for microwave on
        
        '''
        #check if the input is valid
        if not method in ['max elements', 'filenumber', 'elena']:
            print('invalid normalization method, use None instead')
            return 1
        
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            elif 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) < 1717106400:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]
            
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]
            
        #dataframes which contain the parts of self.evs that have microwave on/off
        df_off = self.evs[self.evs['microwave'] == 'off']
        df_on = self.evs[self.evs['microwave'] == 'on']
            
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                t_off[i] = [val for bal in t_off[i] for val in bal]
                h_off[i] = [val for bal in h_off[i] for val in bal]
                t_on[i] = [val for bal in t_on[i] for val in bal]
                h_on[i] = [val for bal in h_on[i] for val in bal]

            data_off = [[[val,kal] for val,kal in zip(t_off[0],h_off[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[1],h_off[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[2],h_off[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_off[3],h_off[3]) if minvol <= kal <= maxvol]]

            data_on = [[[val,kal] for val,kal in zip(t_on[0],h_on[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[1],h_on[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[2],h_on[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_on[3],h_on[3]) if minvol <= kal <= maxvol]]
        
        #if tof is False, we just have to account for trange and vrange
        else:        
            data_off = [[[val,kal] for val,kal in zip([tal for nal in df_off['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel

            data_on = [[[val,kal] for val,kal in zip([tal for nal in df_on['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel
        
        #normalize with the 'max elements' method
        if method == 'max elements':
            h_off = [val for bal in df_off['height_ch1'] if bal != 'None' for val in bal] + [val for bal in df_off['height_ch2'] if bal != 'None' for val in bal] + [val for bal in df_off['height_ch3'] if bal != 'None' for val in bal] + [val for bal in df_off['height_ch4'] if bal != 'None' for val in bal]
            h_on = [val for bal in df_on['height_ch1'] if bal != 'None' for val in bal] + [val for bal in df_on['height_ch2'] if bal != 'None' for val in bal] + [val for bal in df_on['height_ch3'] if bal != 'None' for val in bal] + [val for bal in df_on['height_ch4'] if bal != 'None' for val in bal]
            
            #look how many peaks capped out the voltage
            ma = max(h_off + h_on) #max voltage
            adj = len([val for val in h_off if val == ma]) / len([val for val in h_on if val == ma]) #ratio of the amount of max voltage peaks (mw off/mw on)
                    
        #normalize with the 'filenumber' method
        if method == 'filenumber':
            #dataframes which contain the parts of self.evs that have microwave on/off
            df_off = self.evs[self.evs['microwave'] == 'off']
            df_on = self.evs[self.evs['microwave'] == 'on']
            
            #just take the ratio len(df_off)/len(df_on) where we have to make sure that the LyA column is not empty ('None')
            df_off = df_off[df_off['LyA'] != 'None']
            df_on = df_on[df_on['LyA'] != 'None']
            
            adj = len(df_off) / len(df_on) #ratio of the amount of files
              
        #normalize with the 'elena' method
        if method == 'elena':
            #dataframes which contain the parts of self.evs that have microwave on/off
            df_off = self.evs[self.evs['microwave'] == 'off']
            df_on = self.evs[self.evs['microwave'] == 'on']
            
            #make sure we have a LyA file for each row
            df_off = df_off[df_off['LyA'] != 'None']
            df_on = df_on[df_on['LyA'] != 'None']
            
            #get the LyAdata24 file to get the NE50 line intensities
            data = read_df()
            
            #go through each file for mw off and fetch the Ne50_I value from data
            NE50_off = 0
            for i in range(len(df_off)):
                curr = df_off.iloc[i] #current row of df_off
                NE50_off += float(data[data['LyA'] == curr['LyA']]['NE50_I'])
                
            #repeat for mw on
            NE50_on = 0
            for i in range(len(df_on)):
                curr = df_on.iloc[i] #current row of df_off
                NE50_on += float(data[data['LyA'] == curr['LyA']]['NE50_I'])
                
            adj = NE50_off / NE50_on #ratio of the amount of particles
        
        return adj
    
    
    def volt_hist_sort(self, bins, vrange = None, trange = None, ycut = False, norm = None, tof = False, save = False):
        '''
        get how many peaks are in the bins sorted by voltage and mw with which a histogram can be created
        the histograms are made with the total number of peaks, so if mw on or off cont have similar number of peaks that will be visible
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        vrange = the voltage range in which we want to look at the peak
            input as a list with vrange[0] < vrange[1], default is [0.005, 0.1453334428369999 ]
            the measured voltage only goes up to 0.1453334428369999, so all peaks that would be higher are just exactly at this voltage
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002]
        ycut: defauls False, if True the dont display the height of the last bin fully (only if vrange goes all the way to 0.146 V)
        norm: Choose one of 'None', 'max elements', 'filenumber', 'elena'
            None: No normalization
            max elements: we look at how many peaks mw on/off have at the maximum peak height and multiply the amount of peaks in the bins for mw on by their ration (off/on)
            filenumber: count the amount of files for mw on/off and multiply the amount of peaks in the bins for mw on by their ration (off/on)
            elena: count for each file the amount of particles in the NE50 line and multiply the amount of peaks in the bins for mw on by their ration (off/on)
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        save: if save is not False, the figures gets saved at /eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/, the name of the file is whatever the string of save is
        
        Returns
        ------------
        res = how many events are in each bin
        
        '''
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            elif 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) < 1717106400:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]
                        
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        #binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                t_off[i] = [val for bal in t_off[i] for val in bal]
                h_off[i] = [val for bal in h_off[i] for val in bal]
                t_on[i] = [val for bal in t_on[i] for val in bal]
                h_on[i] = [val for bal in h_on[i] for val in bal]

            data_off = [[[val,kal] for val,kal in zip(t_off[0],h_off[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[1],h_off[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[2],h_off[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_off[3],h_off[3]) if minvol <= kal <= maxvol]]

            data_on = [[[val,kal] for val,kal in zip(t_on[0],h_on[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[1],h_on[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[2],h_on[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_on[3],h_on[3]) if minvol <= kal <= maxvol]]
        
        #if tof is False, we just have to account for trange and vrange
        else:
            #dataframes which contain the parts of self.evs that have microwave on/off
            df_off = self.evs[self.evs['microwave'] == 'off']
            df_on = self.evs[self.evs['microwave'] == 'on']
        
            data_off = [[[val,kal] for val,kal in zip([tal for nal in df_off['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel

            data_on = [[[val,kal] for val,kal in zip([tal for nal in df_on['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel

        height_off = [[val[1] for val in data_off[0]],[val[1] for val in data_off[1]],[val[1] for val in data_off[2]],[val[1] for val in data_off[3]]]
        height_on = [[val[1] for val in data_on[0]],[val[1] for val in data_on[1]],[val[1] for val in data_on[2]],[val[1] for val in data_on[3]]]

        res = [[list(np.histogram(height_off[0], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[1], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[2], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[3], bins = bins, range = (minvol, maxvol))[0])]
               ,[list(np.histogram(height_on[0], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[1], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[2], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[3], bins = bins, range = (minvol, maxvol))[0])]]
        
        ranges = [1000 * val for val in ranges]
        binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        if norm != None:
            adj = self.norm_method(method = norm, vrange = vrange, trange = trange, tof = tof)
            
            #multiply each element in res by the ratio of the amount of max voltage peaks
            #go through each channel
            for i in range(4): 
                #go through each bin
                for k in range(bins):
                    res[1][i][k] = res[1][i][k] * adj  
                    
        xbar = np.zeros(2*bins + 2)
        for i in range(bins + 1):
            for k in range(2):
                xbar[2*i+k] = ranges[i]
        
        fig = plt.figure(layout="constrained", figsize = (50, 40))
        gs = GridSpec(4, 12, figure=fig)
        
        ax1 = fig.add_subplot(gs[0,:3])
        ax2 = fig.add_subplot(gs[0,3:6])
        ax3 = fig.add_subplot(gs[0,6:9])
        ax4 = fig.add_subplot(gs[0,9:])
        ax5 = fig.add_subplot(gs[1, 0:3])
        ax6 = fig.add_subplot(gs[1, 3:6])
        ax7 = fig.add_subplot(gs[1, 6:9])
        ax8 = fig.add_subplot(gs[1, 9:])
        ax9 = fig.add_subplot(gs[2, 1:11])
        ax10 = fig.add_subplot(gs[3, 1:11])
        
        #plot a histogram for each channel
        for i, ax in enumerate(fig.axes[:4]):
            ax.bar([val + 1/2 * binwidth for val in ranges[:-1]], res[0][i], label = 'mw off', align = 'center', width = binwidth, color = '#1f77b4')
            ax.stairs(res[1][i], ranges, color = '#ff7f0e', linewidth = 4, label = 'mw on')
        
        '''
        #plot bar histogram of the datawith mw off
        for i in range(4):
            ax5.bar([val + (i+1)*binwidth/5 for val in ranges[:-1]], [val for val in res[0][i]], width = -0.2 * binwidth, label = 'Ch' + str(i + 1), align = 'center')
        
        #plot bar histogram of the data with mw on
        for i in range(4):
            ax6.bar([val + (i+1)*binwidth/5 for val in ranges[:-1]], [val for val in res[1][i]], width = -0.2 * binwidth, label = 'Ch' + str(i + 1), align = 'center')
        '''
        
        for i, ax in enumerate(fig.axes[4:8]):
            ax.bar([val + 1/2 * binwidth for val in ranges[:-1]], [val - bal for val,bal in zip(res[0][i],res[1][i])], width = 0.9 * binwidth, label = 'difference between mw off/on', align = 'center')
                
        #plot average over the channels of both
        ah = [[(a + b + c + d) for a,b,c,d in zip(res[0][0],res[0][1],res[0][2],res[0][3])],[(a + b + c + d) for a,b,c,d in zip(res[1][0],res[1][1],res[1][2],res[1][3])]]
        ax9.bar([val + 1/2 * binwidth for val in ranges[:-1]], ah[0], width = binwidth, label = 'mw off')
        ax9.stairs(ah[1], ranges, color = '#ff7f0e', linewidth = 4, label = 'mw on')
        
        #plot the difference between the peak distribution
        ax10.bar([val + 1/2 * binwidth for val in ranges[:-1]],[a - b for a,b in zip(ah[0],ah[1])], width = 0.8 * binwidth, label = 'difference between mw on/off', align = 'center')
        
        titles = ['Number of pulses with mw on/off for channel 1','Number of pulses with mw on/off for channel 2','Number of pulses with mw on/off for channel 3','Number of pulses with mw on/off for channel 4',
                  'Microwave on subtracted from microwave off for channel 1', 'Microwave on subtracted from microwave off for channel 2', 'Microwave on subtracted from microwave off for channel 3', 'Microwave on subtracted from microwave off for channel 4',
                  'Number of pulses in all channels with microwave off/on', 'Microwave on subtracted from microwave off']
        
        if ycut == True:
            ycuts = [max(res[0][0][:-1] + res[1][0][:-1]) + 0.2*max(res[0][0][:-1] + res[1][0][:-1]), max(res[0][1][:-1] + res[1][1][:-1]) + 0.2*max(res[0][1][:-1] + res[1][1][:-1]), 
                    max(res[0][2][:-1] + res[1][2][:-1]) + 0.2*max(res[0][2][:-1] + res[1][2][:-1]), max(res[0][3][:-1] + res[1][3][:-1]) + 0.2*max(res[0][3][:-1] + res[1][3][:-1]), 
                    max(res[0][0][:-1] + res[0][1][:-1] + res[0][2][:-1] + res[0][3][:-1]) + 0.2*max(res[0][0][:-1] + res[0][1][:-1] + res[0][2][:-1] + res[0][3][:-1]),
                    max(res[1][0][:-1] + res[1][1][:-1] + res[1][2][:-1] + res[1][3][:-1]) + 0.2*max(res[1][0][:-1] + res[1][1][:-1] + res[1][2][:-1] + res[1][3][:-1]),
                    0, max(ah[0][:-1] + ah[1][:-1]) + 0.2*max(ah[0][:-1] + ah[1][:-1]), 0]
                        
        for i, ax in enumerate(fig.axes):
            for k in ranges:
                ax.axvline(k, linestyle = '--', color = 'black', alpha = 0.2)
            
            #if ycut is true set the ylimits
            if ycut == True and maxvol >= 0.1453334428369999 and not (i == 6 or i == 8):
                ax.set_ylim([0, ycuts[i]])
                
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)
            ax.tick_params(axis = 'x', length = 10)
            
            ax.set_title(label = titles[i], fontsize = 25)
            ax.set_xlabel('Pulse height [mV]', fontsize = 20, labelpad = 20)
            ax.set_ylabel('number of peaks', fontsize = 20, labelpad = 20)
            ax.legend(loc = 'best', fontsize = 15)
        
        if tof:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins. Only peaks within the time of flight window are included.', fontsize = 27, y = 1.03, x = 0.5)
            
        else:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins.', fontsize = 27, y = 1.03, x = 0.5)

        #if save is true we save all figures
        if save != False:
            fig.patch.set_alpha(1.0)
            fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/' + str(save), dpi = 400)
            #fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/' + str(save), dpi = 400, bbox_inches = mtransforms.Bbox([[0.07, 0], [0.43, 0.33]]).transformed(fig.transFigure - fig.dpi_scale_trans),)
            
        return res
            
        
    
    
    
    def tof_peaks(self):
        '''
        
        get the peaks which are within their respective time of flight windows from self.evs separated by microwave off and on
        
        Returns
        ------------
        t_off: time of the peaks which are in the tof window with microwave off
        h_off: heights of the peaks which are in the tof window with microwave off
        t_on: time of the peaks which are in the tof window with microwave on
        h_on: heights of the peaks which are in the tof window with microwave on
        
        '''
        df_off = self.evs[self.evs['microwave'] == 'off']
        df_on = self.evs[self.evs['microwave'] == 'on']
        
        wav = read_df()
        t_off = [[],[],[],[]]
        h_off = [[],[],[],[]]
        t_on = [[],[],[],[]]
        h_on = [[],[],[],[]]
        #go through the 4 channels
        for i in range(4):
            #for mw off
            curr_t_off = list(df_off['time_ch' + str(i+1)]) #times of the peaks for the current channel
            curr_h_off = list(df_off['height_ch' + str(i+1)]) #heights of the peaks for the current channel
            curr_t_off = [val if val != 'None' else [] for val in curr_t_off] #we put [] for any file where there are no peaks for the current channel
            curr_h_off = [val if val != 'None' else [] for val in curr_h_off] #we put [] for any file where there are no peaks for the current channel

            #for mw on
            curr_t_on = list(df_on['time_ch' + str(i+1)]) #times of the peaks for the current channel
            curr_h_on = list(df_on['height_ch' + str(i+1)]) #heights of the peaks for the current channel
            curr_t_on = [val if val != 'None' else [] for val in curr_t_on] #we put [] for any file where there are no peaks for the current channel
            curr_h_on = [val if val != 'None' else [] for val in curr_h_on] #we put [] for any file where there are no peaks for the current channel

            #track on which number of on/off file we currently are in the for loop below
            off_pos = 0
            on_pos = 0

            #go through all the files
            for k in range(len(self.evs)):
                if str(self.evs.iloc[k]['LyA']) != 'None':
                    curr_w = wav[wav['LyA'] == self.evs.iloc[k]['LyA']] #current row from the datafile
                    curr_start = float(curr_w['beam_start']) #beam start at the MCP5
                    curr_stop = float(curr_w['beam_stop']) #beam stop at the MCP5
                    if curr_start == curr_stop: 
                        curr_start, curr_stop = -1,-1
                    else: 
                        curr_start += - 5e-6 + 3e-7 + tof_start #beam start at the LyA detector (8.6e-7 accounts for the travel time from the start of the detector to MCP5)
                        curr_stop += - 5e-6 + 3e-7 + tof_stop #beam stop at the LyA detector (7.0e-7 accounts for the travel time from the end of the detector to MCP5)
                        
                else:
                    curr_start, curr_stop = -1,-1

                #we need to look if for the current file the mw is off or on
                if list(curr_w['microwave'])[0] == 'off': #if mw is off for the current file
                    curr_h_off[off_pos] = [val for val,bal in zip(curr_h_off[off_pos],curr_t_off[off_pos]) if curr_start <= Levs.conv(float(bal)) <= curr_stop] #only keep height during the tof window
                    curr_t_off[off_pos] = [val for val in curr_t_off[off_pos] if curr_start <= Levs.conv(float(val)) <= curr_stop] #only keep time during the tof window
                    off_pos += 1

                if list(curr_w['microwave'])[0] == 'on': #if mw is on for the current file
                    curr_h_on[on_pos] = [val for val,bal in zip(curr_h_on[on_pos],curr_t_on[on_pos]) if curr_start <= Levs.conv(float(bal)) <= curr_stop] #only keep height during the tof window
                    curr_t_on[on_pos] = [val for val in curr_t_on[on_pos] if curr_start <= Levs.conv(float(val)) <= curr_stop] #only keep time during the tof window
                    on_pos += 1
                    
            t_off[i] = curr_t_off
            h_off[i] = curr_h_off
            t_on[i] = curr_t_on
            h_on[i] = curr_h_on
        
        return t_off, h_off, t_on, h_on
    
    
    #get a histogram of the voltage peaks of the 4 channels
    #plot a histrogram with a number of bins corresponding to different voltages at the peaks
    def volt_hist(self, bins, vrange = None, tof = False):
        '''
        get how many peaks are in the bins sorted by voltage with which a histogram can be created
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        vrange = the voltage range in which we want to look at the peak
            input as a list with ranges[0] < ranges[1], default is [0.005, 0.145]
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        
        Returns
        ------------
        res = how many events are in each bin
        
        '''
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            elif 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) < 1717106400:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            heights = [[],[],[],[]]
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data and put all the peak heights into heights
            for i in range(4):
                heights[i] = [val for bal in h_off[i] + h_on[i] for val in bal if minvol <= val <= maxvol]
        
        #if tof is false, just check that the voltages are between minvol and maxvol
        else:
            heights = [[val for bal in [kal for kal in self.evs['height_ch1'] if str(kal) != 'None'] for val in bal if minvol <= val <= maxvol], [val for bal in [kal for kal in self.evs['height_ch2'] if str(kal) != 'None'] for val in bal if minvol <= val <= maxvol], 
                       [val for bal in [kal for kal in self.evs['height_ch3'] if str(kal) != 'None'] for val in bal if minvol <= val <= maxvol], [val for bal in [kal for kal in self.evs['height_ch4'] if str(kal) != 'None'] for val in bal if minvol <= val <= maxvol]] #list with heights of peaks for each channel
        
        #res is how many peaks are in each bin for each channel
        res = [list(np.histogram(heights[0], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(heights[1], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(heights[2], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(heights[3], bins = bins, range = (minvol, maxvol))[0])]
        
        
        #bar histogram with the data
        fig = plt.figure(layout = 'tight', figsize = (40,15))
        gs = GridSpec(2, 4, figure = fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[0,3])
        ax5 = fig.add_subplot(gs[1,:2])
        ax6 = fig.add_subplot(gs[1,2:])
        xticks = [round(val,2) for val in np.linspace(minvol, maxvol, 10)]
        
        #ax1-4
        for i, ax in enumerate(fig.axes[:4]):
            ax.hist([heights[i]], bins, edgecolor = 'black', linewidth = 1)
            ax.set_title('Pulse height histogram of the %ith channel' % i, fontsize = 15)
        
        #ax5
        ax5.hist([heights[0], heights[1], heights[2], heights[3]], ranges, label = ['Ch1', 'Ch2', 'Ch3', 'Ch4'])
        ax5.set_title(label = 'Amount of pulses in the voltage ranges'  + ' separated into ' + str(bins) + ' bins', fontsize = 15, pad = 15)
        ax5.legend(loc = 'upper right')
        
        #ax6:
        ax6.hist([heights[0] + heights[1] + heights[2] + heights[3]], bins, edgecolor = 'black', linewidth = 2)
        ax6.set_title('Amount of pulses in the %i voltage bins' % bins, fontsize = 15)
        
        for ax in fig.axes:
            for i in ranges:
                ax.set_xlabel('Pulse height [V]', fontsize = 15, labelpad = 20)
                ax.set_ylabel('Number of pulses', fontsize = 15, labelpad = 20)
                ax.axvline(i, linestyle = '--', color = 'black', alpha = 0.2)
            ax.set_xticks(ticks = xticks)
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)

        return res
    
    
    def sorted_hist(self, bins, vrange = None, trange = None, tof = False, ycut = False):
        '''
        get how many peaks are in the bins sorted by voltage with which a histogram can be created
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        vrange = the voltage range in which we want to look at the peak
            input as a list with ranges[0] < ranges[1], default is [0.005, 0.145]
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002]
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        ycut: defauls False, if True the dont display the height of the last bin fully (only if vrange goes all the way to 0.146 V)
        
        Returns
        ------------
        res = how many events are in each bin
        
        '''
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            elif 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) < 1717106400:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]
                        
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        #binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                t_off[i] = [val for bal in t_off[i] for val in bal]
                h_off[i] = [val for bal in h_off[i] for val in bal]
                t_on[i] = [val for bal in t_on[i] for val in bal]
                h_on[i] = [val for bal in h_on[i] for val in bal]

            data_off = [[[val,kal] for val,kal in zip(t_off[0],h_off[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[1],h_off[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_off[2],h_off[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_off[3],h_off[3]) if minvol <= kal <= maxvol]]

            data_on = [[[val,kal] for val,kal in zip(t_on[0],h_on[0]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[1],h_on[1]) if minvol <= kal <= maxvol],
                       [[val,kal] for val,kal in zip(t_on[2],h_on[2]) if minvol <= kal <= maxvol],
                        [[val,kal] for val,kal in zip(t_on[3],h_on[3]) if minvol <= kal <= maxvol]]
        
        #if tof is False, we just have to account for trange and vrange
        else:
            #dataframes which contain the parts of self.evs that have microwave on/off
            df_off = self.evs[self.evs['microwave'] == 'off']
            df_on = self.evs[self.evs['microwave'] == 'on']
        
            data_off = [[[val,kal] for val,kal in zip([tal for nal in df_off['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                       [[val,kal] for val,kal in zip([tal for nal in df_off['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_off['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel

            data_on = [[[val,kal] for val,kal in zip([tal for nal in df_on['time_ch1'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch1'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch2'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch2'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch3'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch3'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)], 
                      [[val,kal] for val,kal in zip([tal for nal in df_on['time_ch4'] if str(nal) != 'None' for tal in nal],[tal for nal in df_on['height_ch4'] if str(nal) != 'None' for tal in nal]) if (tmin <= val <= tmax and minvol <= kal <= maxvol)]] #list with times of peaks for each channel

        height_off = [[val[1] for val in data_off[0]],[val[1] for val in data_off[1]],[val[1] for val in data_off[2]],[val[1] for val in data_off[3]]]
        height_on = [[val[1] for val in data_on[0]],[val[1] for val in data_on[1]],[val[1] for val in data_on[2]],[val[1] for val in data_on[3]]]

        res = [[list(np.histogram(height_off[0], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[1], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[2], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_off[3], bins = bins, range = (minvol, maxvol))[0])]
               ,[list(np.histogram(height_on[0], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[1], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[2], bins = bins, range = (minvol, maxvol))[0]), list(np.histogram(height_on[3], bins = bins, range = (minvol, maxvol))[0])]]
        
        binwidth = ranges[1] - ranges[0] #how wide a bin is        
        
        #bar histogram with the data
        fig = plt.figure(layout = 'tight', figsize = (40,25))
        gs = GridSpec(3, 4, figure = fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[0,3])
        ax5 = fig.add_subplot(gs[1,0])
        ax6 = fig.add_subplot(gs[1,1])
        ax7 = fig.add_subplot(gs[1,2])
        ax8 = fig.add_subplot(gs[1,3])
        ax9 = fig.add_subplot(gs[2,:2])
        ax10 = fig.add_subplot(gs[2,2:])
        xticks = [round(val,2) for val in np.linspace(minvol, maxvol, 10)]
        
        #ax1-4
        for i, ax in enumerate(fig.axes[:4]):
            ax.hist([height_off[i]], bins, edgecolor = 'black', linewidth = 1)
            ax.set_title('Pulse height histogram of the %ith channel for microwave off' % i, fontsize = 20)
            
        #ax5-8
        for i, ax in enumerate(fig.axes[4:8]):
            ax.hist([height_on[i]], bins, edgecolor = 'black', linewidth = 1)
            ax.set_title('Pulse height histogram of the %ith channel for microwave on' % i, fontsize = 20)
        
        #ax10
        ax9.hist([height_off[0] + height_off[1] + height_off[2] + height_off[3]], bins, edgecolor = 'black', linewidth = 2)
        ax9.set_title('Amount of pulses in the %i voltage bins for microwave off' % bins, fontsize = 20)
        
        #ax10
        ax10.hist([height_on[0] + height_on[1] + height_on[2] + height_on[3]], bins, edgecolor = 'black', linewidth = 2)
        ax10.set_title('Amount of pulses in the %i voltage bins for microwave on' % bins, fontsize = 20)
        
        if ycut == True:
            ycuts = [max(res[0][0][:-1]) + 0.2*max(res[0][0][:-1]), max(res[0][1][:-1]) + 0.2*max(res[0][1][:-1]), max(res[0][2][:-1]) + 0.2*max(res[0][2][:-1]), max(res[0][3][:-1]) + 0.2*max(res[0][3][:-1]),
                    max(res[1][0][:-1]) + 0.2*max(res[1][0][:-1]), max(res[1][1][:-1]) + 0.2*max(res[1][1][:-1]), max(res[1][2][:-1]) + 0.2*max(res[1][2][:-1]), max(res[1][3][:-1]) + 0.2*max(res[1][3][:-1]),
                    max([val+bal+cal+kal for val,bal,cal,kal in zip(res[0][0][:-1],res[0][1][:-1],res[0][2][:-1],res[0][3][:-1])]) + 0.2*max([val+bal+cal+kal for val,bal,cal,kal in zip(res[0][0][:-1],res[0][1][:-1],res[0][2][:-1],res[0][3][:-1])]),
                    max([val+bal+cal+kal for val,bal,cal,kal in zip(res[1][0][:-1],res[1][1][:-1],res[1][2][:-1],res[1][3][:-1])]) + 0.2*max([val+bal+cal+kal for val,bal,cal,kal in zip(res[1][0][:-1],res[1][1][:-1],res[1][2][:-1],res[1][3][:-1])])]
        
        for k, ax in enumerate(fig.axes):
            for i in ranges:
                ax.set_xlabel('Pulse height [V]', fontsize = 15, labelpad = 20)
                ax.set_ylabel('Number of pulses', fontsize = 15, labelpad = 20)
                ax.axvline(i, linestyle = '--', color = 'black', alpha = 0.2)
                
            ax.set_xticks(ticks = xticks)
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)
            
            if ycut == True:
                ax.set_ylim([0, ycuts[k]])

        return res
        
        
    #plot when the peaks were recorded for a day for the different channels
    #we make all the values of the voltages of the peaks positive here for simplicity
    def peak_distr(self, trange = None, bins = 50, tof = False, save = False):
        '''
        
        plot the distribution of all the peaks when they happened during each recording (each recording lasts 10 microseconds)
        also plot a histogram with the peak distribution for each peak and one for all peaks
          
        Parameters
        ------------
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        cut_high = at this timestamp a vertical line is displayed (except if cut_high = 10002)
        bins = the number of bins in the histograms
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
        save: if save is not False, the figure gets saved at /eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA peak distribution/, the name of the file is whatever the string of save is
        
        '''
        if type(trange) == type(None): cut_low, cut_high = 0, 10002
        else: cut_low, cut_high = trange[0], trange[1]
                       
        ch1 = [val for bal in self.evs['time_ch1'] for val in bal if str(bal) !='None']
        ch2 = [val for bal in self.evs['time_ch2'] for val in bal if str(bal) !='None']
        ch3 = [val for bal in self.evs['time_ch3'] for val in bal if str(bal) !='None']
        ch4 = [val for bal in self.evs['time_ch4'] for val in bal if str(bal) !='None']  
        ch = [ch1, ch2, ch3, ch4]
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            tch = [[],[],[],[]]
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                tch[i] = [val for bal in t_off[i] + t_on[i] for val in bal]
                
            tch1, tch2, tch3, tch4 = tch[0], tch[1], tch[2], tch[3]
        
        colors =['#1f77b4', '#ff7f0e']
        fig = plt.figure(layout = 'tight', figsize = (20,20))
        gs = GridSpec(7, 2, figure = fig)
        ax1 = fig.add_subplot(gs[:3,0])
        ax2 = fig.add_subplot(gs[:3,1])
        ax3 = fig.add_subplot(gs[3:5,0])
        ax4 = fig.add_subplot(gs[3:5,1])
        ax5 = fig.add_subplot(gs[5:,0])
        ax6 = fig.add_subplot(gs[5:,1])
        
        #set the bin ranges from cut_low to cut_high with bins number of bins
        ranges = np.linspace(cut_low, cut_high, bins + 1)
        
        ax1.get_yaxis().set_visible(False)
        ax1.spines[['top', 'bottom', 'left', 'right']].set_alpha(0.4)
        ax1.set_xlabel(xlabel = 'timesteps')
        ax1.set_xlim([0, 10000])
        
        ax1.scatter(ch1, [0.8] * len(ch1), label = 'Ch1 all peaks')
        ax1.scatter(ch2, [0.6] * len(ch2), label = 'Ch2 all peaks')
        ax1.scatter(ch3, [0.4] * len(ch3), label = 'Ch3 all peaks')
        ax1.scatter(ch4, [0.2] * len(ch4), label = 'Ch4 all peaks')
        
        if tof:
            ax1.scatter(tch1, [0.75] * len(tch1), label = 'Ch1 peaks within tof window')
            ax1.scatter(tch2, [0.55] * len(tch2), label = 'Ch2 peaks within tof window')
            ax1.scatter(tch3, [0.35] * len(tch3), label = 'Ch3 peaks within tof window')
            ax1.scatter(tch4, [0.15] * len(tch4), label = 'Ch4 peaks within tof window')   
            
        if (cut_low != 0): ax1.axvline(x = cut_low, linestyle = '--', color = 'black', linewidth = 0.7)
        if (cut_high < 10000): ax1.axvline(x = cut_high, linestyle = '--', label = 'cutoffs', color = 'black', linewidth = 0.7)
        ax1.set_title(label = 'Times when an event was detected for the four channels')
        ax1.legend(loc = 'upper left')
        
        ax2.spines[['top', 'bottom', 'left', 'right']].set_alpha(0.4)
        ax2.set_xlabel(xlabel = 'timesteps')
        ax2.set_ylabel(ylabel = 'number of peaks')
        if (cut_low >= 2000 and cut_high <= 8000): 
            ax2.set_xlim([2000, 8000])
            ranges2 = np.linspace(2000, 8000, bins + 1)
        else: 
            ax2.set_xlim([2000, 8000])
            ranges2 = np.linspace(0, 10002, bins + 1)
        ax2.hist(ch[0]+ch[1]+ch[2]+ch[3], bins = ranges2, edgecolor = 'black', color = colors[0], linewidth = 1, label = 'All peaks')
        if tof:
            ax2.hist(tch[0]+tch[1]+tch[2]+tch[3], bins = ranges2, edgecolor = colors[1], histtype = 'step', linewidth = 2, label = 'Peaks within tof window')
            ax2.set_title(label = 'Histogram of the peak distribution over time for all peaks and for the peaks within the tof window.')
            ax2.legend(loc = 'upper right')
        else:
            ax2.set_title(label = 'Histogram of the peak distribution over time for all peaks')
        if (cut_low != 0): ax2.axvline(x = cut_low, color = 'black', linewidth = 2)
        if (cut_high < 10000): ax2.axvline(x = cut_high, label = 'cutoffs', color = 'black', linewidth = 2)
        
        ch1 = [val for val in ch1 if cut_low < val < cut_high]
        ch2 = [val for val in ch2 if cut_low < val < cut_high]
        ch3 = [val for val in ch3 if cut_low < val < cut_high]
        ch4 = [val for val in ch4 if cut_low < val < cut_high]
        ch = [ch1, ch2, ch3, ch4]
        
        if tof:
            tch1 = [val for val in tch1 if cut_low < val < cut_high]
            tch2 = [val for val in tch2 if cut_low < val < cut_high]
            tch3 = [val for val in tch3 if cut_low < val < cut_high]
            tch4 = [val for val in tch4 if cut_low < val < cut_high]
            tch = [tch1, tch2, tch3, tch4]
                    
        for i, ax in enumerate(fig.axes[2:]):
            if (cut_low > 1000 and cut_high < 9000): ax.set_xlim([cut_low - 20, cut_high + 20])
            else: ax.set_xlim([0, 10002])
            #ax.get_yaxis().set_visible(False)
            ax.spines[['top', 'bottom', 'left', 'right']].set_alpha(0.4)
            ax.set_xlabel(xlabel = 'timesteps')
            ax.set_ylabel(ylabel = 'number of peaks')
            ax.hist(ch[i], bins = ranges, edgecolor = 'black', linewidth = 1, color = colors[0], label = 'All peaks')
            
            if tof:
                ax.hist(tch[i], bins = ranges, edgecolor = colors[1], histtype = 'step', linewidth = 2, label = 'Peaks within tof window')
                ax.set_title(label = 'Histogam of the peak distribution over time for channel %i for all peaks and for the peaks within the tof window.' % (i + 1))
                ax.legend(loc = 'upper right')
            else:
                ax.set_title(label = 'Histogam of the peak distribution over time for channel %i.' % (i + 1))
                
        if save != False:
            fig.patch.set_alpha(1.0)
            fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA peak distribution/' + str(save), dpi = 400)
            
        return


tof_start = - 8.6e-7 #time we have to subtract from the time in the MCP5 waveform where the beam starts passing through the LyA detectors
tof_stop = - 7.0e-7 #time we have to subtract from the time in the MCP5 waveform where the beam stops passing through the LyA detectors

class Levs_sc:
    '''
    
    Analyze the data for the peaks data
    
    Parameters
    ------------
    The input should be a dataframe containing the time and height of peaks as well as their channel and microwave status and corresponding LyA file.
    Use the files generated with Ldate for the correct information
    
    
    '''
    
    #define instance method
    def __init__(self, evs, lim = 0):
        self.evs = evs.reset_index(drop = True)
        datafile = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
        fl = self.evs['LyA']
        fl = [val if val != 'None' else 'empty' for val in fl]
        self.datafile = datafile[[True if val in fl else False for val in datafile.LyA]]
        self.sc_f = [0]*len(fl)
        
        for i in range(len(fl)):
            if fl[i] != 'empty':
                temp_df = self.datafile[self.datafile.LyA == fl[i]]
                if int(temp_df.sc_pow) == -3: self.sc_f[i] = int(temp_df.sc_freq)
            else: self.sc_f[i] = -1 #set sc_f[i] to -1 if there is no LyA file for the current event 
                
        self.f_set = sorted(list(set(self.sc_f)))
        
        if self.f_set[0] == -1:
            self.f_set.pop(0)
        
        if type(lim) == list:
            temp = [0]*len(lim)
            for i in range(len(lim)):
                temp[i] = self.f_set[lim[i]]
            self.f_set = temp
        elif lim != 0:
            self.f_set = self.f_set[:lim]
        
        if type(lim) == int: self.lim = [lim]
        else: self.lim = lim
                    
        self.df = [0]*len(self.f_set)
        
        for i in range(len(self.f_set)):
            self.df[i] = self.evs[[True if val == self.f_set[i] else False for val in self.sc_f]]
            
        self.dff = self.evs[[True if val in self.f_set else False for val in self.sc_f]].copy()
        
        
    #convert the timestep to the actual time (from 0 - 10002 to -5e-6 - 5e-6)
    def conv(i):
        ts = -4.999745000028291e-06
        tf = 5.001254717124112e-06
        tw = tf - ts
        l = 10001
        dt = tw/l
        return ts + dt * i
    
    #convert the actual time to the timestamp(from -5e-6 - 5e-6 to 0 - 10002)
    def conv2(i):
        ts = -4.999745000028291e-06
        tf = 5.001254717124112e-06
        tw = tf - ts
        l = 10001
        dt = tw/l
        return (i - ts) / dt
    
    
    def tof_peaks(self):
        '''
        
        get the peaks which are within their respective time of flight windows from self.evs separated by microwave off and on
        
        Returns
        ------------
        t_off: time of the peaks which are in the tof window with microwave off
        h_off: heights of the peaks which are in the tof window with microwave off
        t_on: time of the peaks which are in the tof window with microwave on
        h_on: heights of the peaks which are in the tof window with microwave on
        
        '''
        df_sc = self.evs
        
        wav = read_df()
        t_sc = [[],[],[],[]]
        h_sc = [[],[],[],[]]
        #go through the 4 channels
        for i in range(4):
            curr_t_sc = list(df_sc['time_ch' + str(i+1)]) #times of the peaks for the current channel
            curr_h_sc = list(df_sc['height_ch' + str(i+1)]) #heights of the peaks for the current channel
            curr_t_sc = [val if val != 'None' else [] for val in curr_t_sc] #we put [] for any file where there are no peaks for the current channel
            curr_h_sc = [val if val != 'None' else [] for val in curr_h_sc] #we put [] for any file where there are no peaks for the current channel
            
            sc_pos = 0
            
            #go through all the files
            for k in range(len(self.evs)):
                if str(self.evs.iloc[k]['LyA']) != 'None':
                    curr_w = wav[wav['LyA'] == self.evs.iloc[k]['LyA']] #current row from the datafile
                    #print(len(curr_w))
                    curr_start = float(curr_w['beam_start']) #beam start at the MCP5
                    curr_stop = float(curr_w['beam_stop']) #beam stop at the MCP5
                    if curr_start == curr_stop: 
                        curr_start, curr_stop = -1,-1
                    else: 
                        curr_start += - 5e-6 + 3e-7 + tof_start #beam start at the LyA detector (8.6e-7 accounts for the travel time from the start of the detector to MCP5)
                        curr_stop += - 5e-6 + 3e-7 + tof_stop #beam stop at the LyA detector (7.0e-7 accounts for the travel time from the end of the detector to MCP5)
                        
                else:
                    curr_start, curr_stop = -1,-1

                curr_h_sc[sc_pos] = [val for val,bal in zip(curr_h_sc[sc_pos],curr_t_sc[sc_pos]) if curr_start <= Levs.conv(float(bal)) <= curr_stop] #only keep height during the tof window
                curr_t_sc[sc_pos] = [val for val in curr_t_sc[sc_pos] if curr_start <= Levs.conv(float(val)) <= curr_stop] #only keep time during the tof window
                sc_pos += 1

                    
            t_sc[i] = curr_t_sc
            h_sc[i] = curr_h_sc
        
        return t_sc, h_sc
    
    
    def mw_scan(self, bins, plot = True, vrange = None, trange = None, tof = False, ycut = False, save = False, norm = None):
        '''
        get how many peaks are in the bins sorted by voltage and different microwave scanner frequencies
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        plot: default True, if false then res will be returned and no plots will be generated
        vrange = the voltage range in which we want to look at the peak
            input as a list with ranges[0] < ranges[1], default is [0.005, 0.145]
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002]
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        ycut: defauls False, if True the dont display the height of the last bin fully (only if vrange goes all the way to 0.146 V)
        save: if save is not False, the figures gets saved at /eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/scans, the name of the file is whatever the string of save is
        norm: Choose one of 'None', 'max elements', 'filenumber', 'elena'
            None: No normalization
            high_pulses: we look at how many peaks each frequency has above 0.175 V and normalize them using this number
            filenumber: count the amount of files for each frequency and normalize the different frequencies with respect to these numbers
            elena: count for each file the amount of particles in the NE50 line and normalize the different frequencies with respect to this count
            density: divide each bin by the total number of pulses for the corresponding frequency, we get a probability distribution with this method
            density_ch: same as density but now seperately for each channel, so the output has now shape (len(f_set), 4), so res has to be calculated differently
        
        Returns
        ------------
        res = how many events are in each bin
        
        '''     
        if vrange == None: minvol, maxvol = 0.005, 0.2906665578484535
        else: minvol, maxvol = vrange[0], vrange[1]
                        
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            ts, hs = self.tof_peaks()
        
        #if tof is False, we just have to account for trange and vrange
        else:            
            ts = [[],[],[],[]]
            hs = [[],[],[],[]]
            for i in range(4):
                ts[i] = [val if val != 'None' else [] for val in list(self.evs['time_ch' + str(i+1)])] 
                hs[i] = [val if val != 'None' else [] for val in list(self.evs['height_ch' + str(i+1)])]
        
        datf = [[]]*len(self.f_set)
        
        for i in range(len(self.f_set)):
            t_temp_arr = [[],[],[],[]]
            h_temp_arr = [[],[],[],[]]
            temp_arr = [[],[],[],[]]
            for k in range(4):
                #d
                t_temp_arr[k] = [val for val,bal in zip(ts[k], self.sc_f) if bal == self.f_set[i]]
                h_temp_arr[k] = [val for val,bal in zip(hs[k], self.sc_f) if bal == self.f_set[i]]
                temp_arr[k] = [[val,bal] for kal,zal in zip(t_temp_arr[k],h_temp_arr[k]) for val,bal in zip(kal,zal)]
                temp_arr[k] = [val for val in temp_arr[k] if (tmin<=val[0]<=tmax and minvol<=val[1]<=maxvol)]
            datf[i] = temp_arr
        
        height = [[]]*len(self.f_set)
        
        for i in range(len(self.f_set)):
            temp_arr = [[],[],[],[]]
            for k in range(4):
                temp_arr[k] = [val[1] for val in datf[i][k]]
            height[i] = temp_arr
            
        if norm == None:
            adj = [1]*len(self.f_set)
                
        else:
            adj = self.norm_method(norm)   
        
        if norm == 'density_ch' or norm == 'density_ch_tof' or norm == 'off_calibration':
            res = [[]]*len(self.f_set)
            
            for i in range(len(self.f_set)):
                temp_arr = [[],[],[],[]]
                for k in range(4):
                    temp_arr[k] = list(np.histogram(height[i][k], bins = bins, range = (minvol, maxvol))[0])
                    temp_arr[k] = [val*adj[i][k] for val in temp_arr[k]]
                res[i] = temp_arr
        
        else:
            res = [[]]*len(self.f_set)

            for i in range(len(self.f_set)):
                temp_arr = [[],[],[],[]]
                for k in range(4):
                    temp_arr[k] = list(np.histogram(height[i][k], bins = bins, range = (minvol, maxvol))[0])
                    temp_arr[k] = [val*adj[i] for val in temp_arr[k]]
                res[i] = temp_arr
            
        if plot == False: return res
         
        binwidth = ranges[1] - ranges[0] #how wide a bin is        
        
        fig = plt.figure(layout="constrained", figsize = (30, 25))
        gs = GridSpec(3, 4, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        ax5 = fig.add_subplot(gs[2, 0:4])
        
        #plot a histogram for each channel
        for i, ax in enumerate(fig.axes[:4]):
            for k in range(len(self.f_set)):
                if k == 0:
                    ax.bar([val + 1/2 * binwidth for val in ranges[:-1]], res[k][i], align = 'center', width = binwidth)
                
                ax.stairs(res[k][i], ranges, linewidth = 6.5-0.5*k, label = str(self.f_set[k]/1000000000) + ' GHz')
                        
        #plot average over the channels
        for k in range(len(self.f_set)):
            ah = [(a + b + c + d) for a,b,c,d in zip(res[k][0],res[k][1],res[k][2],res[k][3])]
            if k == 0 and True:
                ax5.bar([val + 1/2 * binwidth for val in ranges[:-1]], [val for val in ah], width = binwidth)
            
            ax5.stairs(ah, ranges, linewidth = 6.5-0.5*k, label = str(self.f_set[k]/1000000000) + ' GHz')
                             
        titles = ['Number of pulses with different mw scanner frequencies for channel 1', 'Number of pulses with different mw scanner frequencies for channel 2', 
                  'Number of pulses with different mw scanner frequencies for channel 3', 'Number of pulses with different mw scanner frequencies for channel 4', 
                  'Number of pulses in all channels with different mw scanner frequencies']
        
        if ycut == True:
            ycuts = [max([val for bal in res for val in bal[0][:-1]]) * 1.1, max([val for bal in res for val in bal[1][:-1]]) * 1.1,
                    max([val for bal in res for val in bal[2][:-1]]) * 1.1, max([val for bal in res for val in bal[3][:-1]]) * 1.1, 
                    max([val+bal+kal+zal for tal in res for val,bal,kal,zal in zip(tal[0][:-1],tal[1][:-1],tal[2][:-1],tal[3][:-1])]) * 1.1]

                                
        for i, ax in enumerate(fig.axes):
            for k in ranges:
                ax.axvline(k, linestyle = '--', color = 'black', alpha = 0.2)
                
            if ycut == True:
                ax.set_ylim([0, ycuts[i]])
            
            if i == 4: ax.set_xticks(ticks = np.linspace(minvol, maxvol, 59))
            else: ax.set_xticks(ticks = np.linspace(minvol, maxvol, 30))
            #if i == 4: ax.set_xticks(ticks = np.linspace(minvol, maxvol, 50))
            #else: ax.set_xticks(ticks = np.linspace(minvol, maxvol, 30))
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)
            ax.tick_params(axis = 'x', length = 10)
            
            ax.set_title(label = titles[i], fontsize = 25)
            ax.set_xlabel('Pulse height [mV]', fontsize = 20, labelpad = 20)
            ax.set_ylabel('number of peaks', fontsize = 20, labelpad = 20)
            ax.legend(loc = 'upper left', fontsize = 15)
        
        if tof:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins. Only peaks within the time of flight window are included.', fontsize = 27, y = 1.03, x = 0.5)
            
        else:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins.', fontsize = 27, y = 1.03, x = 0.5)
        
        plt.show()
        
        #if save is true we save all figures
        if save != False:
            fig.patch.set_alpha(1.0)
            fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/' + str(save), dpi = 400)
            #fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA histograms/' + str(save), dpi = 400, bbox_inches = mtransforms.Bbox([[0.07, 0], [0.43, 0.33]]).transformed(fig.transFigure - fig.dpi_scale_trans),)
            
        return res
    
    
    
    
    
    def scatter_ph_t(self, trange = [0, 10001]):
        '''
        
        2d scatter plot with the pulse height on the y-axis and the time of the pulse on the x-axis
        
        Parameters
        ------------
        trange: default [0, 10001], the timesteps between which we want to see the plot
        
        '''
        ts = [[],[],[],[]]
        hs = [[],[],[],[]]
        
        for i in range(4):
            ts[i] = [val if val != 'None' else [] for val in list(self.dff['time_ch' + str(i+1)])]
            hs[i] = [val if val != 'None' else [] for val in list(self.dff['height_ch' + str(i+1)])]
            
            ts[i] = [val for bal in ts[i] for val in bal]
            hs[i] = [val for bal in hs[i] for val in bal]
        
        fig = plt.figure(layout="constrained", figsize = (30, 30))
        gs = GridSpec(3, 4, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        
        #scatter plot for the pulses from each channel seperately
        for i, ax in enumerate(fig.axes):
            ax.scatter(ts[i], hs[i], label = 'Pulse height', s = 12)
            ax.set_title(label = 'Channel ' + str(i), fontsize = 25, pad = 10)
            
        #scatterplot for all pulses
        ax5 = fig.add_subplot(gs[2, 1:3])
        ta = ts[0]+ts[1]+ts[2]+ts[3]
        ha = hs[0]+hs[1]+hs[2]+hs[3]
        ax5.scatter(ta, ha, label = 'Pulse height', s = 4)
        ax5.set_title(label = 'All Channels', fontsize = 25, pad = 10)
        
        #format changes for all 5 subplots
        for i, ax in enumerate(fig.axes):
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_xlim(trange[0], trange[1])
            ax.set_xlabel('Timestep', fontsize = 20)
            ax.set_ylabel('Pulse height [mV]', fontsize = 20)
            #lgd = ax.legend(loc = 'upper right', fontsize = 20)
            #lgd.legend_handles[0]._sizes = [40]
        
        return
    
    
    def hist2d_ph_t(self, bins, trange = [0, 10002], vrange = [0.005, 0.2906665578484536], colormap = 'hot'):
        '''
        
        2d histogram with the pulse height on the y-axis and the time of the pulse on the x-axis
        
        Parameters
        ------------
        bins: array of length 2, the number of bins we want for the time (bins[0]) and the pulse height (bins[1])
        trange: default [0, 10001], the timesteps between which we want to see the histogram
        vrange: default [0.005, 0.2906665578484536], pulse height between which pulses are included
        colormap: default 'hot', define which colormap from matplotlib should be used
        
        '''        
        ts = [[],[],[],[]]
        hs = [[],[],[],[]]
        
        for i in range(4):
            ts[i] += [val if val != 'None' else [] for val in list(self.dff['time_ch' + str(i+1)])]
            hs[i] += [val if val != 'None' else [] for val in list(self.dff['height_ch' + str(i+1)])]

            ts[i] = [val for bal in ts[i] for val in bal]
            hs[i] = [val for bal in hs[i] for val in bal]

            hs[i] = [bal for val,bal in zip(ts[i], hs[i]) if trange[0]<=val<=trange[1]]
            ts[i] = [val for val in ts[i] if trange[0]<=val<=trange[1]]
            
        res0, et0, eh0 = np.histogram2d(ts[0], hs[0], bins = bins, range = [trange, vrange])
        res1, et1, eh1 = np.histogram2d(ts[1], hs[1], bins = bins, range = [trange, vrange])
        res2, et2, eh2 = np.histogram2d(ts[2], hs[2], bins = bins, range = [trange, vrange])
        res3, et3, eh3 = np.histogram2d(ts[3], hs[3], bins = bins, range = [trange, vrange])
        
        res = [res0, res1, res2, res3, res0+res1+res2+res3] #matrix with the 2d histogram values
        et = [et0, et1, et2, et3] #values of the edges of the time bins, has length bins[0]+1
        eh = [eh0, eh1, eh2, eh3] #values of the edges of the height bins, has length bins[1]+1
        
        fig = plt.figure(layout="constrained", figsize = (30, 30))
        gs = GridSpec(3, 4, figure=fig)
        cr = [trange[0], trange[1], vrange[0], vrange[1]] #range of the colormap of the current plot (left, right, bottom, top)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        ax5 = fig.add_subplot(gs[2, :2])
        
        #plot the histogram for the pulses in each channel seperately
        for i, ax in enumerate(fig.axes):
            cmax = max([val for bal in res[i] for val in bal[:-1]]) #max value for the colormap
            
            temp_mat = np.flip(res[i].transpose(), 0).copy()
            
            temp_ax = ax.imshow(temp_mat, cmap = colormap, vmin = 0, vmax = cmax, aspect = (trange[1]-trange[0])*2.35, extent = cr)
            temp_cbar = plt.colorbar(temp_ax, ax = ax, location = 'right', orientation = 'vertical', pad = 0.01, fraction = 0.03, aspect = 25, use_gridspec = True, ticks = [0,max([val for bal in res[i] for val in bal])])
            temp_cbar.ax.tick_params(labelsize = 15)
            
            if i == 4: ax.set_title(label = 'All Channels', fontsize = 25, pad = 10)
            else: ax.set_title(label = 'Channel ' + str(i), fontsize = 25, pad = 10)
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_xlabel('Timestep', fontsize = 20, labelpad = 20)
            ax.set_ylabel('Pulse height [mV]', fontsize = 20, labelpad = 20)
        
        return
    
    
    def scatter_ph_e(self):
        '''
        
        2d scatter plot with the pulse height on the y-axis and the event number on the x-axis
        
        '''
        hs = [[],[],[],[]]
        ev = [[],[],[],[]]
        
        for i in range(4):
            hs[i] = [val if val != 'None' else [] for val in list(self.dff['height_ch' + str(i+1)])]
            ev[i] = [[val]*len(bal) for val,bal in zip(range(len(hs[i])), hs[i])]
            
            hs[i] = [val for bal in hs[i] for val in bal]
            ev[i] = [val for bal in ev[i] for val in bal]
        
        fig = plt.figure(layout="constrained", figsize = (30, 30))
        gs = GridSpec(3, 4, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        
        #scatter plot for the pulses from each channel seperately
        for i, ax in enumerate(fig.axes):
            ax.scatter(ev[i], hs[i], label = 'Pulse height', s = 12)
            ax.set_title(label = 'Channel ' + str(i), fontsize = 25, pad = 10)
            
        #scatterplot for all pulses
        ax5 = fig.add_subplot(gs[2, 1:3])
        ta = ev[0]+ev[1]+ev[2]+ev[3]
        ha = hs[0]+hs[1]+hs[2]+hs[3]
        ax5.scatter(ta, ha, label = 'Pulse height', s = 4)
        ax5.set_title(label = 'All Channels', fontsize = 25, pad = 10)
        
        #format changes for all 5 subplots
        for i, ax in enumerate(fig.axes):
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_xlabel('Event number', fontsize = 20)
            ax.set_ylabel('Pulse height [mV]', fontsize = 20)
            #lgd = ax.legend(loc = 'upper right', fontsize = 20)
            #lgd.legend_handles[0]._sizes = [40]
        
        return
    
    
    def hist2d_ph_e(self, bins, trange = [0, 10002], vrange = [0.005, 0.2906665578484536], colormap = 'hot'):
        '''
        
        2d histogram with the pulse height on the y-axis and the event number on the x-axis
        
        Parameters
        ------------
        bins: array of length 2, the number of bins we want for the time (bins[0]) and the pulse height (bins[1])
        trange: default [0, 10001], the timesteps between which pulses have to be to be included
        vrange: default [0.005, 0.2906665578484536], pulse height between which pulses are included
        colormap: default 'hot', define which colormap from matplotlib should be used
        
        '''
        ts = [[],[],[],[]]
        hs = [[],[],[],[]]
        ev = [[],[],[],[]]
        
        for i in range(4):
            ts[i] += [val if val != 'None' else [] for val in list(self.dff['time_ch' + str(i+1)])]
            hs[i] += [val if val != 'None' else [] for val in list(self.dff['height_ch' + str(i+1)])]
            ev[i] = [[val]*len(bal) for val,bal in zip(range(len(hs[i])), hs[i])]

            ts[i] = [val for bal in ts[i] for val in bal]
            hs[i] = [val for bal in hs[i] for val in bal]
            ev[i] = [val for bal in ev[i] for val in bal]

            hs[i] = [bal for val,bal in zip(ts[i], hs[i]) if trange[0]<=val<=trange[1]]
            ev[i] = [bal for val,bal in zip(ts[i], ev[i]) if trange[0]<=val<=trange[1]]
            
        res0, ee0, eh0 = np.histogram2d(ev[0], hs[0], bins = bins, range = [[0, len(self.dff)-1], vrange])
        res1, ee1, eh1 = np.histogram2d(ev[1], hs[1], bins = bins, range = [[0, len(self.dff)-1], vrange])
        res2, ee2, eh2 = np.histogram2d(ev[2], hs[2], bins = bins, range = [[0, len(self.dff)-1], vrange])
        res3, ee3, eh3 = np.histogram2d(ev[3], hs[3], bins = bins, range = [[0, len(self.dff)-1], vrange])
        
        res = [res0, res1, res2, res3, res0+res1+res2+res3] #matrix with the 2d histogram values
        ee = [ee0, ee1, ee2, ee3] #values of the edges of the time bins, has length bins[0]+1
        eh = [eh0, eh1, eh2, eh3] #values of the edges of the height bins, has length bins[1]+1
        
        fig = plt.figure(layout="constrained", figsize = (30, 30))
        gs = GridSpec(3, 4, figure=fig)
        cr = [0, len(self.dff)-1, vrange[0], vrange[1]] #range of the colormap of the current plot (left, right, bottom, top)
        
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        ax5 = fig.add_subplot(gs[2, :2])
        
        colorbars = [0, 0, 0, 0]
                
        #plot the histogram for the pulses in each channel seperately
        for i, ax in enumerate(fig.axes):
            cmax = max([val for bal in res[i] for val in bal[:-1]]) #max value for the colormap
            
            temp_mat = np.flip(res[i].transpose(), 0).copy()
            
            temp_ax = ax.imshow(temp_mat, cmap = colormap, vmin = 0, vmax = cmax, aspect = 2.25*len(self.dff), extent = cr)
            temp_cbar = plt.colorbar(temp_ax, ax = ax, location = 'right', orientation = 'vertical', pad = 0.01, fraction = 0.0258, aspect = 25, use_gridspec = True, ticks = [0,max([val for bal in res[i] for val in bal])])
            temp_cbar.ax.tick_params(labelsize = 15)
            
            if i == 4: ax.set_title(label = 'All Channels', fontsize = 25, pad = 10)
            else: ax.set_title(label = 'Channel ' + str(i), fontsize = 25, pad = 10)
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_xlabel('Timestep', fontsize = 20, labelpad = 20)
            ax.set_ylabel('Pulse height [mV]', fontsize = 20, labelpad = 20)
        
        return
    
    
    def norm_method(self, method):
        '''
        
        get the value to adjust the microwave on bar heights for different normalization methods
        
        Parameters
        ------------
        method: the method we want to use for the normalization
            high_pulses: we look at how many peaks each frequency has above 0.175 V and normalize them using this number
            filenumber: count the amount of files for each frequency and normalize the different frequencies with respect to these numbers
            elena: count for each file the amount of particles in the NE50 line and normalize the different frequencies with respect to this count
            density: divide each bin by the total number of pulses for the corresponding frequency, we get a probability distribution with this method
            density_ch: same as density but now seperately for each channel, so the output has now shape (len(f_set), 4)
            
        Returns
        ------------
        adj: a list which has the amount of different frequencies as length which contains floats (one for each scanner frequency) that should be multiplied with each bin height for microwave on
        
        '''
        #check if the input is valid
        if not method in ['high_pulses', 'filenumber', 'elena', 'density', 'density_ch_tof', 'density_ch', 'off_calibration']:
            print('invalid normalization method, use None instead')
            return [1]*len(self.f_set)
            
        maxvol = 0.2906665578484535
        
        if method == 'high_pulses':
            hp = [0]*len(self.f_set)
            for i in range(len(self.f_set)):
                h_ch1 = [val if val != 'None' else [] for val in self.df[i]['height_ch1']]
                h_ch2 = [val if val != 'None' else [] for val in self.df[i]['height_ch2']]
                h_ch3 = [val if val != 'None' else [] for val in self.df[i]['height_ch3']]
                h_ch4 = [val if val != 'None' else [] for val in self.df[i]['height_ch4']]
                currh = [val+bal+kal+zal for val,bal,kal,zal in zip(h_ch1, h_ch2, h_ch3, h_ch4)]
                currh = [val for bal in currh for val in bal]
                hp[i] = len([1 for val in currh if val >= 0.8*maxvol])
            adj = [hp[1]/val for val in hp]
                               
        #normalize with the 'filenumber' method
        if method == 'filenumber':
            fn = [0]*len(self.f_set)
            for i in range(len(self.f_set)):
                curr_t1 = [val if val != 'None' else [] for val in self.df[i]['time_ch1']]
                curr_t2 = [val if val != 'None' else [] for val in self.df[i]['time_ch2']]
                curr_t3 = [val if val != 'None' else [] for val in self.df[i]['time_ch3']]
                curr_t4 = [val if val != 'None' else [] for val in self.df[i]['time_ch4']]
                fn[i] = len(self.df[i][[True if len(val+bal+kal+zal)>0 else False for val,bal,kal,zal in zip(curr_t1, curr_t2, curr_t3, curr_t4)]])
                #fn[i] = len(self.df[i])
            adj = [fn[1]/val for val in fn]
              
        #normalize with the 'elena' method
        if method == 'elena':
            el = [0]*len(self.f_set)
            for i in range(len(self.f_set)):
                curr_t1 = [val if val != 'None' else [] for val in self.df[i]['time_ch1']]
                curr_t2 = [val if val != 'None' else [] for val in self.df[i]['time_ch2']]
                curr_t3 = [val if val != 'None' else [] for val in self.df[i]['time_ch3']]
                curr_t4 = [val if val != 'None' else [] for val in self.df[i]['time_ch4']]
                #el[i] = sum(self.df[i]['NE50_I'])
                el[i] = sum([val for val,bal,kal,zal,tal in zip(self.evs['NE50_I'], curr_t1, curr_t2, curr_t3, curr_t4) if len(bal+kal+zal+tal)>0])
            adj = [el[1]/val for val in el]
            
        if method == 'density':
            den = [0]*len(self.f_set)
            for i in range(len(self.f_set)):
                curr_t1 = [val if val != 'None' else [] for val in self.df[i]['time_ch1']]
                curr_t2 = [val if val != 'None' else [] for val in self.df[i]['time_ch2']]
                curr_t3 = [val if val != 'None' else [] for val in self.df[i]['time_ch3']]
                curr_t4 = [val if val != 'None' else [] for val in self.df[i]['time_ch4']]
                den[i] = sum([len(val+bal+kal+zal) for val,bal,kal,zal in zip(curr_t1, curr_t2, curr_t3, curr_t4)])
            adj = [1/val for val in den]
            
        if method == 'density_ch':
            den = [[0]*4]*len(self.f_set)
            adj = [[0]*4]*len(self.f_set)
            for i in range(len(self.f_set)):
                curr_t1 = [val if val != 'None' else [] for val in self.df[i]['time_ch1']]
                curr_t2 = [val if val != 'None' else [] for val in self.df[i]['time_ch2']]
                curr_t3 = [val if val != 'None' else [] for val in self.df[i]['time_ch3']]
                curr_t4 = [val if val != 'None' else [] for val in self.df[i]['time_ch4']]
                den[i] = [sum([len(val) for val in curr_t1]), sum([len(val) for val in curr_t2]), sum([len(val) for val in curr_t3]), sum([len(val) for val in curr_t4])]
                adj[i] = [1/den[i][0], 1/den[i][1], 1/den[i][2], 1/den[i][3]]
                
        if method == 'off_calibration':
            if not 0 in set(self.sc_f):
                print('Microwave off not in the dataset')
                return [1]*len(self.f_set)
            cdf = self.evs[[True if val==0 else False for val in self.sc_f]]
            bt = []
            times = list(cdf.Time)
            temp_t = [0,0]
            for i in range(len(cdf)-1):
                if temp_t[0] == 0:
                    temp_t[0] = times[i]
                elif i == len(cdf)-2:
                    temp_t[1] = times[-1]
                    bt += [temp_t]
                elif times[i+1]-times[i] > 100:
                    temp_t[1] = times[i]
                    bt += [temp_t]
                    temp_t = [0,0]
                    
            corr = [[0]*4]*len(bt)
            for i in range(len(bt)):
                tcorr = [0]*4
                tcdf = cdf[[True if bt[i][0]<=val<=bt[i][1] else False for val in cdf.Time]]
                
                tch1 = [val if val!='None' else [] for val in list(tcdf.time_ch1)]
                tch2 = [val if val!='None' else [] for val in list(tcdf.time_ch2)]
                tch3 = [val if val!='None' else [] for val in list(tcdf.time_ch3)]
                tch4 = [val if val!='None' else [] for val in list(tcdf.time_ch4)]
                
                tcorr[0] = len([val for bal in tch1 for val in bal])
                tcorr[1] = len([val for bal in tch2 for val in bal])
                tcorr[2] = len([val for bal in tch3 for val in bal])
                tcorr[3] = len([val for bal in tch4 for val in bal])
                
                corr[i] = [val*len(tcdf)/100 for val in tcorr]
                                
            adj = [[0]*4]*len(self.f_set)
            for i in range(len(self.f_set)):
                temp_arr = []
                temp_len = []
                cf = self.f_set[i]
                pdf = self.df[i]
                for k in range(len(bt)):
                    tpdf = pdf[pdf.Time<bt[0][0]] if k == 0 else pdf[[True if bt[k-1][1]<val<bt[k][0] else False for val in pdf.Time]]
                    if len(tpdf)>0:
                        temp_len += [len(tpdf)]
                        
                        tch1 = [val if val!='None' else [] for val in list(tpdf.time_ch1)]
                        tch2 = [val if val!='None' else [] for val in list(tpdf.time_ch2)]
                        tch3 = [val if val!='None' else [] for val in list(tpdf.time_ch3)]
                        tch4 = [val if val!='None' else [] for val in list(tpdf.time_ch4)]
                        
                        temp_num = [0]*4
                        temp_num[0] = len([val for bal in tch1 for val in bal])
                        temp_num[1] = len([val for bal in tch2 for val in bal])
                        temp_num[2] = len([val for bal in tch3 for val in bal])
                        temp_num[3] = len([val for bal in tch4 for val in bal])
                        
                        temp_arr += [temp_num]
                        #print(len(tpdf),len(pdf))
                        temp_arr[-1] = [len(tpdf)/(len(pdf)*val) for val in corr[k]]
                
                temp_val = [0]*4
                for y in range(4):
                    temp_val[y] = sum([val/len(temp_arr) for val in [val[y] for val in temp_arr]])
                
                adj[i] = temp_val
                
        if method == 'density_ch_tof':
            den = [[0]*4]*len(self.f_set)
            adj = [[0]*4]*len(self.f_set)
            t, h = self.tof_peaks()
            for i in range(len(self.f_set)):
                curr_t1 = [val for val,bal,kal in zip(t[0],h[0],self.sc_f) if kal == self.f_set[i]]
                curr_t2 = [val for val,bal,kal in zip(t[1],h[1],self.sc_f) if kal == self.f_set[i]]
                curr_t3 = [val for val,bal,kal in zip(t[2],h[2],self.sc_f) if kal == self.f_set[i]]
                curr_t4 = [val for val,bal,kal in zip(t[3],h[3],self.sc_f) if kal == self.f_set[i]]
                den[i] = [sum([len(val) for val in curr_t1]), sum([len(val) for val in curr_t2]), sum([len(val) for val in curr_t3]), sum([len(val) for val in curr_t4])]
                adj[i] = [1/den[i][0], 1/den[i][1], 1/den[i][2], 1/den[i][3]]
            
        return adj
