#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
import os
from Lfile import Lfile
from Ldate import Ldate
from gbarDataLoader24 import loadShortSummary
from LyAdata24 import read_df
from tqdm import tqdm
pd.set_option("display.max_rows",90)
pd.set_option("display.max_columns",None)


# In[68]:


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
    
    
    #same as plot_volt_hist but now with 2 histograms, 1 with mw off and 1 with mw on
    def volt_hist_sort(self, bins, vrange = None, trange = None, tof = False):
        '''
        get how many peaks are in the bins sorted by voltage and mw with which a histogram can be created
        the histograms are all normalized
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        vrange = the voltage range in which we want to look at the peak
            input as a list with vrange[0] < vrange[1], default is [0.005, 0.145]
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002]
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        
        Returns
        ------------
        res = how many events are in each bin
        
        '''
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]
            
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                t_off[i] = [val for bal in t_off[i] for val in bal]
                h_off[i] = [val for bal in h_off[i] for val in bal]
                t_on[i] = [val for bal in t_on[i] for val in bal]
                h_on[i] = [val for bal in h_on[i] for val in bal]

            data_off = [[[val,kal] for val,kal in zip(t_off[0],h_off[0])],
                       [[val,kal] for val,kal in zip(t_off[1],h_off[1])],
                       [[val,kal] for val,kal in zip(t_off[2],h_off[2])],
                        [[val,kal] for val,kal in zip(t_off[3],h_off[3])]]

            data_on = [[[val,kal] for val,kal in zip(t_on[0],h_on[0])],
                       [[val,kal] for val,kal in zip(t_on[1],h_on[1])],
                       [[val,kal] for val,kal in zip(t_on[2],h_on[2])],
                        [[val,kal] for val,kal in zip(t_on[3],h_on[3])]]
        
        else:
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
        
        tot_off = sum([val for bal in res[0] for val in bal])
        tot_on = sum([val for bal in res[1] for val in bal])
        
        #we want the values in res to be percentages of the peaks in each bin (the sum of both mw on/off is 1 over the 4 channels)
        for k in range(4):
            res[0][k] = [val/tot_off for val in res[0][k]]
            res[1][k] = [val/tot_on for val in res[1][k]]
        
        fig = plt.figure(layout="constrained", figsize = (50, 30))
        gs = GridSpec(3, 12, figure=fig)
        
        ax1 = fig.add_subplot(gs[0,:3])
        ax2 = fig.add_subplot(gs[0,3:6])
        ax3 = fig.add_subplot(gs[0,6:9])
        ax4 = fig.add_subplot(gs[0,9:])
        ax5 = fig.add_subplot(gs[1, 0:4])
        ax6 = fig.add_subplot(gs[1, 4:8])
        ax7 = fig.add_subplot(gs[1, 8:])
        ax8 = fig.add_subplot(gs[2, 1:5])
        ax9 = fig.add_subplot(gs[2, 6:10])
        
        #plot a histogram for each channel
        for i, ax in enumerate(fig.axes[:4]):
            ax.hist(height_off[i], bins = ranges, label = 'mw off', density = True)
            ax.hist(height_on[i], bins = ranges, histtype = 'step', linewidth = 4, label = 'mw on', density = True)         
            ax.set_ylabel('Peak distribution', fontsize = 20, labelpad = 20)
        
        #plot bar histogram of the datawith mw off
        for i in range(4):
            ax5.bar([val + (i+1)*binwidth/5 for val in ranges[:-1]], [val * 100 for val in res[0][i]], width = -0.2 * binwidth, label = 'Ch' + str(i + 1), align = 'center')
        ax5.set_ylabel('Percentages of peaks', fontsize = 20, labelpad = 20)
        
        #plot bar histogram of the data with mw on
        for i in range(4):
            ax6.bar([val + (i+1)*binwidth/5 for val in ranges[:-1]], [val * 100 for val in res[1][i]], width = -0.2 * binwidth, label = 'Ch' + str(i + 1), align = 'center')
        ax6.set_ylabel('Percentages of peaks', fontsize = 20, labelpad = 20)
        
        #plot difference between mw on/off        
        for i in range(4):
            ax7.bar([val + (i+1)*binwidth/5 for val in ranges[:-1]], [(val - bal) * 100 for val,bal in zip([val / sum(res[0][i]) for val in res[0][i]],[val / sum(res[1][i]) for val in res[1][i]])], width = -0.2 * binwidth, label = 'Ch' + str(i), align = 'center')
        ax7.set_ylabel('Percentages of peaks', fontsize = 20, labelpad = 20)
                    
        #plot average over the channels of both
        ah = [[(a + b + c + d) * 100 for a,b,c,d in zip(res[0][0],res[0][1],res[0][2],res[0][3])],[(a + b + c + d) * 100 for a,b,c,d in zip(res[1][0],res[1][1],res[1][2],res[1][3])]]
        ax8.hist([height_off[0] + height_off[1] + height_off[2] + height_off[3]], bins = ranges, label = 'mw off', density = True, align = 'mid')
        ax8.hist([height_on[0] + height_on[1] + height_on[2] + height_on[3]], bins = ranges, histtype = 'step', linewidth = 5, label = 'mw on', density = True)
        ax8.set_ylabel('Peak distribution', fontsize = 20, labelpad = 20)
        
        #plot the difference between the peak distribution
        ax9.bar([val + 0.5 * binwidth for val in ranges[:-1]],[a - b for a,b in zip(ah[0],ah[1])], width = 0.8 * binwidth, label = 'difference between mw on/off', align = 'center')
        ax9.set_ylabel('Percentages of peaks', fontsize = 20, labelpad = 20)
        
        titles = ['Peak distribution with mw on/off for channel 1','Peak distribution with mw on/off for channel 2','Peak distribution with mw on/off for channel 3','Peak distribution with mw on/off for channel 4',
                  'Percentage of peaks in the voltage ranges with mw off', 'Percentage of peaks in the voltage ranges with mw on', 
                  'Microwave on subtracted from microwave off for the four channels','Peak distribution with microwave off/on averaged over the channels', 
                  'Microwave on subtracted from microwave off']
        
        xticks = [round((val + bal)/2, 4) for val,bal in zip(ranges[:-1],ranges[1:])]
        
                        
        for i, ax in enumerate(fig.axes):
            for k in ranges:
                ax.axvline(k, linestyle = '--', color = 'black', alpha = 0.5)
                
            ax.set_xticks(ticks = xticks)
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)
            ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
            ax.xaxis.get_offset_text().set_fontsize(15)
            ax.tick_params(axis = 'x', length = 10)
            
            ax.set_title(label = titles[i], fontsize = 25)
            ax.set_xlabel('Voltage [V]', fontsize = 20, labelpad = 20)
            ax.legend(loc = 'upper right', fontsize = 15)
            
        if tof:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins with normalized peak number for mw off/on. Only peaks within the time of flight window are included.', fontsize = 27, y = 1.03, x = 0.5)

        
        else:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(round(minvol, 3)) + ' and ' + str(round(maxvol, 3)) + ' separated into ' + str(bins) + ' bins with normalized peak number for mw off/on.', fontsize = 27, y = 1.03, x = 0.5)
            
        return data_off
    
    
    def plot_fit(self, bins, vrange = None, trange = None, dim = 3, tof = False):
        '''
        
        plot polynomials fit to data for each channel with mw on/off
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        h = 0.003 is the minimum height for a peak to be recorded
        vrange = the voltage range in which we want to look at the peak
            input as a list with vrange[0] < vrange[1], default is [0.005, 0.145]
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        dim = dimension of the polynomial 
        tof = False if tof = True, only the peaks within the time of flight of the beam at the LyA detector are used. The tof is calculated by looking at the MCP5 waveform and using the distance to the LyA detector to calculate the tof window.
            if tof = True and trange != None, the trange gets ignored
        
        '''
        if vrange == None:
            if 1716415200 <= int(self.evs[[True if str(val)!='None' else False for val in self.evs.LyA]].iloc[0]['LyA'][-18:-8]) <= 1716587999:
                minvol, maxvol = 0.005, 0.2906665578484535
            else:
                minvol, maxvol = 0.005, 0.1453334428369999 
        else: minvol, maxvol = vrange[0], vrange[1]
            
        if (trange == None or tof == True): tmin, tmax = 0, 10002
        else: tmin, tmax = trange[0], trange[1]

        ranges = np.linspace(minvol, maxvol, num = bins + 1).tolist() #make a list with values separating the bins   
        binwidth = ranges[1] - ranges[0] #how wide a bin is
        
        #if tof is True we only take the peaks which are within their tof window
        if tof:
            t_off, h_off, t_on, h_on = self.tof_peaks()
            
            #go through the 4 channels and flatten the data
            for i in range(4):
                t_off[i] = [val for bal in t_off[i] for val in bal]
                h_off[i] = [val for bal in h_off[i] for val in bal]
                t_on[i] = [val for bal in t_on[i] for val in bal]
                h_on[i] = [val for bal in h_on[i] for val in bal]

            data_off = [[[val,kal] for val,kal in zip(t_off[0],h_off[0])],
                       [[val,kal] for val,kal in zip(t_off[1],h_off[1])],
                       [[val,kal] for val,kal in zip(t_off[2],h_off[2])],
                        [[val,kal] for val,kal in zip(t_off[3],h_off[3])]]

            data_on = [[[val,kal] for val,kal in zip(t_on[0],h_on[0])],
                       [[val,kal] for val,kal in zip(t_on[1],h_on[1])],
                       [[val,kal] for val,kal in zip(t_on[2],h_on[2])],
                        [[val,kal] for val,kal in zip(t_on[3],h_on[3])]]
        
        else:
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
        
        step = ranges[1] - ranges[0] #how wide a bin is
        x_axis = [val * step + minvol + step/2 for val in range(bins)] #x axis value for the scatter plot of number of peaks in the bins
        
        t = np.linspace(minvol, maxvol, 50) #x axis to plot with the fit polynomial
        
        fig = plt.figure(layout="constrained", figsize = (25, 12))
        gs = GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(gs[:2,:2])
        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[0,3])
        ax4 = fig.add_subplot(gs[1,2])
        ax5 = fig.add_subplot(gs[1,3]) 
        
        xticks = np.linspace(minvol, maxvol, num = 10)
        
        off = ax1.hist([height_off[0] + height_off[1] + height_off[2] + height_off[3]], bins = ranges, density = True, label = 'Hist for mw off')
        fit_off = np.poly1d(np.polyfit(x_axis, off[0], dim)) #fit for mw off
        
        on = ax1.hist([height_on[0] + height_on[1] + height_on[2] + height_on[3]], bins = ranges, histtype = 'step', linewidth = 5, density = True, label = 'Hist for mw on')
        fit_on = np.poly1d(np.polyfit(x_axis, on[0], dim)) #fit for mw off
        
        ax1.plot(t, fit_off(t), label = 'Fit for mw off', color = 'blue') #plot the fit for channel i with mw off
        ax1.plot(t, fit_on(t), label = 'Fit for mw on', color = 'red') #plot the fit for channel i with mw on
        
        for k in ranges:
            ax1.axvline(k, linestyle = '--', color = 'black', alpha = 0.5)
        
        ax1.set_xticks(ticks = xticks)
        ax1.tick_params(axis = 'x', labelsize = 10, labelrotation = 45)
        ax1.tick_params(axis = 'y', labelsize = 10)
        ax1.set_xlabel('Voltage [V]', fontsize = 15, labelpad = 15)
        ax1.set_ylabel('Peak density', fontsize = 15, labelpad = 15)
        ax1.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
        ax1.xaxis.get_offset_text().set_fontsize(10)
        ax1.set_title('Peak distribution with mw off/on for all channels', fontsize = 15)
        ax1.legend(loc = 'upper right')
        
        #go through the 4 channels
        for i, ax in enumerate(fig.axes[1:]):
            for k in ranges:
                ax.axvline(k, linestyle = '--', color = 'black', alpha = 0.5)
                
            #fit_off = np.poly1d(np.polyfit(x_axis, [val * 100 for val in [val / sum(res[0][i]) for val in res[0][i]]], dim)) #fit for mw off
            temp_off = ax.hist(height_off[i], bins = ranges, density = True, label = 'Hist for mw off')
            fit_off = np.poly1d(np.polyfit(x_axis, temp_off[0], dim)) #fit for mw off
            #ax.bar(x_axis, [val * 100 for val in [val / sum(res[0][i]) for val in res[0][i]]], width = step, label = 'Percentage of peaks in the voltage ranges with mw off') #scatter plot for channel i with mw off
            
            #fit_on = np.poly1d(np.polyfit(x_axis, [val * 100 for val in [val / sum(res[1][i]) for val in res[1][i]]], dim)) #fit for mw on
            temp_on = ax.hist(height_on[i], bins = ranges, histtype = 'step', linewidth = 4, density = True, label = 'Hist for mw on')
            fit_on = np.poly1d(np.polyfit(x_axis, temp_on[0], dim)) #fit for mw off
            #ax.bar(x_axis, [val * 100 for val in [val / sum(res[1][i]) for val in res[1][i]]], width = step, fill = False, label = 'Percentage of peaks in the voltage ranges with mw on') #scatter plot for channel i with mw on
            
            ax.plot(t, fit_off(t), label = 'Fit for mw off', color = 'blue') #plot the fit for channel i with mw off
            ax.plot(t, fit_on(t), label = 'Fit for mw on', color = 'red') #plot the fit for channel i with mw on
            
            ax.set_xticks(ticks = xticks)
            ax.tick_params(axis = 'x', labelsize = 10, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 10)
            ax.set_xlabel('Voltage [V]', fontsize = 10, labelpad = 15)
            ax.set_ylabel('Peak density', fontsize = 10, labelpad = 15)
            ax.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0,0))
            ax.xaxis.get_offset_text().set_fontsize(10)
            ax.set_title('Peak distribution with mw off/on for channel ' + str(i + 1), fontsize = 13)
            ax.legend(loc = 'upper right')
        
        #change title if tof = True
        if tof:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(minvol) + ' and ' + str(maxvol) + ' separated into ' + str(bins) + ' bins and fits through the peak data. \n Only peaks within the time of flight window are included.', fontsize = 20, y = 1.05, x = 0.5)

        else:
            fig.suptitle('Histograms of the peaks found in the Lyman Alpha MCPs with a voltage between ' + str(minvol) + ' and ' + str(maxvol) + ' separated into ' + str(bins) + ' bins and fits through the peak data.', fontsize = 20, y = 1.02, x = 0.5)
        
        return
    
    
    #same as plot_volt_hist but now with 2 histograms, 1 with mw off and 1 with mw on
    def volt_hist_sort2(self, bins, plot = True, vrange = None, trange = None, ycut = False, norm = None, tof = False, save = False):
        '''
        get how many peaks are in the bins sorted by voltage and mw with which a histogram can be created
        the histograms are made with the total number of peaks, so if mw on or off cont have similar number of peaks that will be visible
        
        Parameters
        ------------
        bins = the number of bins we create the histogram with
        plot: default True, if false then only res will be returned and no plots generated
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
                    
        if plot == False: return res
                    
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
            ycuts = [max(list(res[0][0][:-1])+list(res[1][0][:-1]))*1.1, max(list(res[0][1][:-1])+list(res[1][1][:-1]))*1.1, max(list(res[0][2][:-1])+list(res[1][2][:-1]))*1.1, max(list(res[0][3][:-1])+list(res[1][3][:-1]))*1.1, -1, -1, -1, -1, max(ah[0][:-1]+ah[1][:-1])*1.1, -1]
                        
        for i, ax in enumerate(fig.axes):
            for k in ranges:
                ax.axvline(k, linestyle = '--', color = 'black', alpha = 0.2)
            
            #if ycut is true set the ylimits
            if ycut == True and maxvol >= 0.1453334428369999 and ycuts[i]>=0:
                ax.set_ylim([0, ycuts[i]])
                
            ax.tick_params(axis = 'x', labelsize = 15, labelrotation = 45)
            ax.tick_params(axis = 'y', labelsize = 15)
            ax.tick_params(axis = 'x', length = 10)
            if i == 8: ax.set_xlim([1000*(minvol)-0.6, 1000*(maxvol)+0.6])
            elif i == 9: ax.set_xlim([1000*(minvol)-0.4, 1000*(maxvol)+0.4])
            else: ax.set_xlim([1000*(minvol)-4, 1000*(maxvol)+4])
            
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
        if trange == None: cut_low, cut_high = 0, 10002
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
        


# In[ ]:




