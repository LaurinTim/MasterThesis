#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
import scipy.optimize as scio


# In[4]:


def sinfunc(x,A,w,p,c):
    '''
    
    sinus function which we can fit to the data
    
    Parameters
    ------------
    x: variable of the function
    A: amplitude of the sinus wave
    w: adjusting the sin frequency
    p: horizontal offset of the sin
    c: vertical offset of the sin
    
    Returns
    ------------
    value of the sinus functions given by the parameters at point x
    
    '''
    return A*np.sin(w*x + p) + c


def sinfit(xx, yy):
    '''
    
    get a sin fit for the data
    we fit through the datapoints 8000 to 10000 since we expect peaks between 4000 and 6000 and before that there is a lot of additional noise
    
    Parameters
    ------------
    xx: x coordinates of the data
    yy: y coordinates of the data
    
    Returns
    ------------
    fitfunc: fit sinus function to xx and yy
    
    '''
    xx = np.array(xx)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(xx), (xx[1]-xx[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    popt, pcov = scio.curve_fit(sinfunc, xx, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda x: A * np.sin(w*x + p) + c
    return fitfunc, A


# In[54]:


#class for 1 trc file with data from the lyman alpha detectors
class Lfile():
    '''
    
    Class for a single trc file which gives info like the peak locations and heights and whether the mw was off/on for the measurement
    
    Parameters
    ------------
    path of the file
    
    '''
    
    def __init__(self, filepath, df = None):
        self.filepath = filepath
        if type(df) != type(None):
            self.df = df
        else:
            self.df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
        
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
        run = list(self.df[self.df['LyA'] == self.filepath]['run'])[0]
        
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
        
        mw_power = list(self.df[self.df['LyA'] == self.filepath]['MW_power'])[0]
        
        if (mw_power > 0.0001): #take the third channel because it is most visible there
            return 1
        else:
            return 0
    
    
    def fit_file(self):
        '''

        get the sin fit functions for the 4 channels in a list

        Returns
        ------------
        func: list with the 4 sinus fits to the data of the file

        '''
        data = self.read_trc()
        yy = data[1]
        xx = np.linspace(0,10001,10002)
        ret = [0,0,0,0]
        func = [0,0,0,0] #for the fits of the 4 channels
        amp = [0,0,0,0] #for the amplitudes of the 4 channels

        for i in range(4):
            try:
                ret[i] = sinfit(xx[8000:10000], yy[i][8000:10000])
            except:
                func[i] = lambda x: 0
                amp[i] = 0
            else:
                func[i] = ret[i][0]
                amp[i] = ret[i][1]

        return func, amp
    
    
    #plot the data from the ith channel (channel 5 for all of them added, this is the default)
    #keep in mind that the peaks for channel 5 are where the peaks for the single channels are, so it might look weird
    #use channel 6 to plot all 4 channels in 1 plot
    def plot_voltage(self,p = 0.001, h = 0.005, d = 10, back = 30, trange = None, comb = False, no_cor = False, show_fit = False, tof = False, save = False):
        '''
        plot the voltage recorded for the channels corrected by a sin fit
        
        Parameters
        ------------
        p = 0.001 is a parameter used to determine the peaks
        h = 0.005 is the minimum height for a peak to be recorded
        d = 10 the distance two peaks have to be apart from eachother
        back = 30 how many datapoints we go back to determine the std, with which we determine whether something is a peak or not
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        comb: default False, if True then only a plot with the combined voltages and peaks from the 4 channels is output
        no_cor: default False, if True then also plot the uncorrected voltages, only works with comb = False
        show_fit: default False, if True then plot the sin fit with which we correct the voltages if mw is on and the average if mw is off, only works with comb = False
        tof: default False, if True then show the time of flight window we get from the waveform file, only works with comb = False
        save: if save is not False, the figure gets saved at /eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA MCPs images/, the name of the file is whatever the string of save is

        '''
        if trange == None: cut_low, cut_high = 0,10002
        else: cut_low, cut_high = trange[0], trange[1]
        
        data = self.read_trc() #get the data for the file
        t, h, m, r = self.get_events_from_file(prom = p, hgt = h, dist = d, l = back, trange = trange) #get the time and heights for the peaks

        time = data[0][0] #use this for the time
        volt = [data[1][0], data[1][1], data[1][2], data[1][3]] #list with the voltages for the different channels
        time_peaks = [list(t[0]), list(t[1]), list(t[2]), list(t[3])] #list with the times of the peaks for the different channels
        height_peaks = [list(h[0]), list(h[1]), list(h[2]), list(h[3])]
        voltc = [0,0,0,0]
        av = [0,0,0,0]
        mw = m
        
        #fit a sin function to the voltage of the 4 channels and subtract it from the voltage if mw is on
        if mw == 1:
            xl = np.linspace(0, len(time), len(time)+1)
            fitc, ampc = self.fit_file() #list with the sinus fits for the channels
            for i in range(4):
                #only apply the sinfit if the amplitude of the sin is bigger than 0.0015V
                if ampc[i] >= 0.0015:
                    voltc[i] = [val - fitc[i](bal) for val,bal in zip(volt[i],xl)] #plus because volt is the negative of the recorded voltage
                else:
                    av[i] = np.average(volt[i][8000:10000])
                    voltc[i] = [val - av[i] for val in volt[i]]
                     
        #if mw is off, we subtract the average of voltc[8000:10000] from voltc
        else:
            for i in range(4):
                av[i] = np.average(volt[i][8000:10000])
                voltc[i] = [val - av[i] for val in volt[i]]
                
        #if tof is True, we retrieve the values
        if tof == True:
            #get the LyA datafile
            lyadf = read_df()
            
            #get the beam start and stop time for the current file from the waveform file for the date
            currf = lyadf[lyadf['LyA'] == self.filepath]
            window = [float(currf['beam_start']), float(currf['beam_stop'])]
            #convert the time from the waveform file to the LyA file
            window[0] += - 5e-6 + 3e-7 - 8.6e-7
            window[1] += - 5e-6 + 3e-7 - 7.0e-7
                    
        if comb:
            fig = plt.figure(figsize = (15,8))
            
            for i in range(4):
                chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
                plt.scatter(time[cut_low:cut_high], voltc[i][cut_low:cut_high], label = 'Voltage channel ' + str(i))
                plt.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 100)
                
            plt.legend(loc = 'best')
            plt.xlabel(xlabel = 'Time [µs]')
            plt.ylabel(ylabel = 'Voltage [V]')
            plt.title(label = 'Voltages and peaks of the four channels')
            return
        
        fig = plt.figure(layout = 'tight', figsize = (20,10))
        gs = GridSpec(2, 2, figure = fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        
        for i, ax in enumerate(fig.axes):
            chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
            ax.scatter(time[cut_low:cut_high], voltc[i][cut_low:cut_high], s = 15, label = 'Corrected voltage')
            if no_cor:
                ax.scatter(time[cut_low:cut_high], volt[i][cut_low:cut_high], s = 15, label = 'Uncorrected voltage')
                ax.legend(loc = 'upper right')
            if show_fit:
                if mw == 1: 
                    if ampc[i] >= 0.002:
                        ax.plot(np.linspace(time[cut_low],time[cut_high], 2* (cut_high - cut_low)), fitc[i](np.linspace(cut_low,cut_high, 2*(cut_high-cut_low))), color = 'red', label = 'Sinus fit')
                    else:
                        ax.plot([time[cut_low], time[cut_high]], [av[i], av[i]], color = 'red', label = 'Average')
                else: ax.plot([time[cut_low], time[cut_high]], [av[i], av[i]], color = 'red', label = 'Average')
                ax.legend(loc = 'upper right')
            if tof == True:
                ax.axvline(window[0], color = 'black', linestyle = '--')
                ax.axvline(window[1], color = 'black', linestyle = '--')
            ax.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 30, linewidth = 3)
            #ax.grid(True, alpha = 1)
            ax.set_ylabel(ylabel = 'Voltage [V]')
            ax.set_xlabel(xlabel = 'Time [s]')#'Time [µs]')
            ax.set_title(label = 'Voltage and peaks in channel %i' % (i + 1))
        
        #ax1.plot(np.linspace(time[cut_low], time[cut_high], 1001), fitc[0](np.linspace(cut_low, cut_high, 1001)))
        #ax1.scatter(time[cut_low:cut_high], [val - fitc[0](bal) for val,bal in zip(volt[0][cut_low:cut_high],xl[cut_low:cut_high])])
        
        if save != False:
            fig.patch.set_alpha(1.0)
            fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA MCPs images/' + str(save), dpi = 400)
        
        return time_peaks, height_peaks
    
    
    #get events from one file, return list with 4 lists where we can see where in the data of the 4 seperate channels there are peaks and a list with the peak heights
    #mw will return 0 (if the mw is turned off) or 1 (if the mw is turned on)
    def get_events_from_file(self, prom = 0.001, hgt = 0.005, dist = 10, l = 30, trange = None):
        '''
        get information about the events recorded by the 4 mcps in the lya setup for the file
        
        Parameters
        ------------
        prom = 0.001 is a parameter used to determine the peaks
        hgt = 0.005 is the minimum height for a peak to be recorded
        dist = 10 the distance two peaks have to be apart from eachother
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
        volt = [-data[1][0], -data[1][1], -data[1][2], -data[1][3]] #voltage data for the four channels (negative because we need to search for max peaks)
        
        events = [0,0,0,0] #list to put the lists where the events are into   
        p_height = [[],[],[],[]] #list for the peak heights of the 4 detectors
        mw = self.mw_on() #return mw = 1 if mw was on, otherwise mw = 0
        voltc = [0,0,0,0]
        
        #fit a sin function to the voltage of the 4 channels and subtract it from the voltage if mw is on
        if mw == 1:
            xl = np.linspace(0, len(time), len(time)+1)
            fitc, ampc = self.fit_file() #list with the sinus fits for the channels
            for i in range(4):
                #only apply the sinfit if the amplitude of the sin is bigger than 0.0015V
                if ampc[i] > 0.0015:
                    voltc[i] = [val + fitc[i](bal) for val,bal in zip(volt[i],xl)] #plus because volt is the negative of the recorded voltage
                else:
                    av = np.average(volt[i][8000:10000])
                    voltc[i] = [val - av for val in volt[i]]
                     
        #if mw is off, we subtract the average of voltc[8000:10000] from voltc
        else:
            for i in range(4):
                av = np.average(volt[i][8000:10000])
                voltc[i] = [val - av for val in volt[i]]
                
        if 1716933600 <= int(self.filepath[-18:-8]) < 1716588000:
            maxvol = 0.2906665578484535
        elif 1716415200 <= int(self.filepath[-18:-8]) < 1717106400:
            maxvol = 0.2906665578484535
        else:
            maxvol = 0.1453334428369999 
        
        #go through the data of the four detectors
        for i in range (4):
            #get the average for the current channel and subtract it from the voltage
            #avg = np.average(volt[i][8000:10000])
            #volt[i] = [val - avg for val in volt[i]]
            
            peaks = sci.find_peaks(voltc[i], prominence = prom, height = hgt, distance = dist, wlen = 200) #get list of potential peaks, the parameters could be improved
            #if i == 3: print(peaks)
            events[i] = peaks[0] #set the ith element of the list 'events' to the locations of the peaks
            events[i] = [val for val in events[i] if val > cut_low and val < cut_high] #only take peaks in trange
            
            ###################
            ph = -1
            cch = -1
            ###################
            
            if i == cch: 
                print(events[cch])
                print([voltc[cch][val] for val in events[cch]])
                print()
            
            #go through all the potential peaks
            for m in range(len(events[i])):
                #we want to disregard peaks with certain features (we set events[i][m] = -1 for the ones we want to discard) (these criteria have been found through trial and error, i am sure that there are better ones)
                currh = voltc[i][events[i][m]] #height of the current peak
                currt = events[i][m] #timestep of the current peak
                if (l > 10):
                    #look at the voltage elements between 10 and l (default 25) positions before the peak and calculate the standard deviation
                    tl = voltc[i][(currt-l) : (currt-10)] #list with voltage for the datapoints from l before the mth peak of the ith channel until 10 before the peak
                    if len(tl) == 0: tl = [0,1]
                    mean = sum(tl) / len(tl) #mean of the elements in tl
                    variance = sum([((x - mean) ** 2) for x in tl]) / len(tl) #variance of the elements in tl
                    std = variance ** 0.5 #standard deviation of the elements in tl
                    
                    tln = voltc[i][(currt+15) : (currt+l)] #list with voltage for the datapoints from 10 after the mth peak of the ith channel until 30 after the peak
                    if len(tln) == 0: tln = [0,1]
                    meann = sum(tln) / len(tln) #mean of the elements in tln
                    variancen = sum([((x - meann) ** 2) for x in tln]) / len(tln) #variance of the elements in tln
                    stdn = variancen ** 0.5 #standard deviation of the elements in tln
                    
                    #get the average value of the 5 voltages before the peak
                    av = sum([val for val in voltc[i][(events[i][m]-5):events[i][m]]])/5
                    
                    #if currt is in the first 20 elements, we disregard the peak
                    if currt < 2000:
                        events[i][m] = -1
                        currh = 10
                    
                    #we keep all peaks with greater height than 0.05V
                    if currh <= 0.05:
                        #discard all peaks before the timestep 2000 since there is too much noise and that is far away from when the beam travels through the detector
                        
                        #if the standard deviation divided by the peak height is bigger than 0.25 discard the peak
                        #if std/currh > 0.5:
                        #    if i == cch and currt == ph:
                        #        print('a0\n')
                        #    events[i][m] = -1
                            
                        if i == cch and currt == ph:
                            print('output')
                            print(currh)
                            print(std/currh)
                            print(std)
                            print(stdn/currh)
                            print(stdn)
                            print()
                            
                        left = 3
                        if voltc[i][currt-1] == currh: left = 4
                        
                        #check if the difference betwen the peak voltage and 5 elements before the peak is greater than 0.02 or the standard deviation divided by peak height is larger than 0.25
                        if currh - abs(voltc[i][currt-5]) < 0.04 and currh - voltc[i][currt-5] < 0.06 and (currh - voltc[i][currt-left] < currh*0.9 or (std+stdn)/currh > 0.6):
                            if i == cch and currt == ph: print('y')
                            #if the standard deviation divided by the peak height is bigger than 0.15 and the surrounding elements do not fulfill certain conditions we discard the peak 
                            '''try here also the difference if we have 0*std/currh instead of 3*std/currh and if the last criteria is good or bad'''
                            if (events[i][m] != -1 and std/currh > 0.13 and 
                                 (max(voltc[i][currt-20:currt-5]) >= currh/(2+3*std/currh) or min(voltc[i][currt:currt+20]) <= -currh/1.9 or max(voltc[i][currt+10:currt+25]) >= currh/1.5)
                               and max(voltc[i][currt-200:currt]) >= 2*currh):
                                if i == cch and currt == ph:
                                    print('a')
                                    print(currh)
                                    print(std)
                                    print(std/currh)
                                    print(stdn/currh)
                                    #tlt = voltc[i][(currt-4) : (currt+8)]
                                    #meant = sum(tlt) / len(tlt)
                                    #print('test std: %f' % (sum([(val - meant)**2 for val in tlt])) ** 0.5)
                                    print(max(voltc[i][currt-20:currt-5]) >= currh/(2+3*std/currh))
                                    print(min(voltc[i][currt:currt+20]) <= -currh/1.9)
                                    print(max(voltc[i][currt+10:currt+25]) >= currh/1.5)
                                    print()
                                events[i][m] = -1

                            if events[i][m] != -1 and av < 0.002 and currh < 0.04:
                                if i == cch and currt == ph:
                                    print('b\n')
                                events[i][m] = -1
                                
                        #if the peak is smaller than 0.0085V the standard deviation before and after the peak has to be smaller than 0.001 and there has to be no voltage >= currh/3 in the 20 voltages before and after to keep it
                        if events[i][m] != -1 and currh < 0.0086:
                            if i == cch and currt == ph: print('x')
                            if (std > 0.001 or stdn > 0.001) and max([abs(val) for val in voltc[i][currt-20:currt-4]+voltc[i][currt+5:currt+20]]) >= currh/3:
                                if i == cch and currt == ph:
                                    print(std)
                                    print(stdn)
                                    print('c\n')
                                events[i][m] = -1
                                
                        #if the peak is smaller than the voltage 4 elements before the peak plus 1/5 of the peak, discard it
                        if events[i][m] != -1 and currh <= voltc[i][currt-4] + currh/5:
                            if i == cch and currt == ph:
                                print('d\n')
                            events[i][m] = -1
                            
                        #if the peak is smaller than the voltage 4 elements after the peak plus 1/5 of the peak, discard it
                        if events[i][m] != -1 and currh <= voltc[i][currt+6] + currh/5:
                            if i == cch and currt == ph:
                                print('e\n')
                            events[i][m] = -1
                        
                        #if there is a higher voltage than 0.08V in the 30 elements before the peak and peak is smaller than 0.04V, discard it
                        #if (events[i][m] != -1 and max(voltc[i][currt-30:currt]) >= 0.08 and currh <= 0.04): 
                        #    if i == cch and currt == ph:
                        #        print('f\n')
                        #    events[i][m] = -1
                            
                        #if there is a higher voltage than 0.08V in the 200 elements before the peak and peak is smaller than 0.02V, discard it
                        'maybe should be 0.02 or 0.03 instead of 0.015'
                        if (events[i][m] != -1 and max(voltc[i][currt-200:currt]) >= 0.08 and currh <= 0.02 and (std/currh > 0.15 or (stdn + std)/currh > 0.25) and currh - voltc[i][currt-left] < 0.9*currh): 
                            if i == cch and currt == ph:
                                print(std/currh)
                                print(stdn/currh)
                                print((std+stdn)/currh)
                                print('g\n')
                            events[i][m] = -1
                            
                        if (events[i][m] != -1 and max(voltc[i][currt-200:currt]) >= 0.08 and currh <= 0.04 and std/currh > 0.15 and currh < 2*max(voltc[i][currt-20:currt-4] + voltc[i][currt+5:currt+20])): 
                            if i == cch and currt == ph:
                                print(std/currh)
                                print('g1\n')
                            events[i][m] = -1
                            
                        #if there is a voltage element in the 15 points before the peak that is negative and its absolute is bigger than 0.5 of the peak height and there is a voltage greater than 2*currh in the last 200 elements, discard the peak
                        if abs(min(voltc[i][currt-15:currt])) >= currh/2 and max(voltc[i][currt-200:currt]) >= 2*currh:
                            if i == cch and currt == ph:
                                print(currh, abs(min(voltc[i][currt-15:currt])))
                                print('h\n')
                            events[i][m] = -1
                            
                        #if std/currh > 0.4 and the max of the last 30 elements is bigger than 1.2*currh, discard the peak
                        if std/currh > 0.4 and (stdn/currh > 0.15 or max(voltc[i][currt-15:currt]) >= 2*currh) and max(voltc[i][currt-30:currt]) > 1.2*currh:
                            if i == cch and currt == ph:
                                print(std/currh)
                                print(stdn/currh)
                                print('i0\n')
                            events[i][m] = -1
                        
                        if i == cch and currt == ph:
                            print('\nkeep')
                            print(std/currh)
                            print(stdn/currh)
                            print(max([abs(val) for val in voltc[i][currt-20:currt-4]+voltc[i][currt+5:currt+20]]))
                            print(max(voltc[i][currt-20:currt-5]) >= currh/(2+3*std/currh))
                            print(min(voltc[i][currt:currt+20]) <= -currh/2)
                            print(max(voltc[i][currt+10:currt+25]) >= currh/1.5)
                            
                        #if events[i][m] != -1 and std/currh > 0.35:
                        #    print('\nERROR: ' + str(i) + ', ' + str(currt) + ', ' + str(currh) + ', ' + str(std/currh))
                    
                    #if there is a peak that is bigger than maxvol/2 in the last 30 voltages, discard any peak smaller than maxvol/3.5
                    if (events[i][m] != -1 and max(voltc[i][currt-30:currt]) >= maxvol/2 and currh <= maxvol/3.5):
                        if i == cch and currt == ph:
                            print('i1\n')
                        events[i][m] = -1
                        
                    #if there is a peak that is bigger than maxvol/3 in the last 20 voltages, discard any peak smaller than maxvol/6
                    if (events[i][m] != -1 and max(voltc[i][currt-20:currt]) >= maxvol/3 and currh <= maxvol/6):
                        if i == cch and currt == ph:
                            print(maxvol/3, maxvol/6)
                            print('i3\n')
                        events[i][m] = -1
                    
                else:
                    if (volt[i][events[i][m]] < hgt):
                        events[i][m] = -1

            events[i] = [val for val in events[i] if val != -1] #only keep events that are not at position -1
            
            #if the peak height in volt is maxvol or the peak height in voltc is bigger than maxvol, we put maxvol as the peak height, otherwise the voltc value of the peak
            for b in range(len(events[i])):
                if (volt[i][events[i][b]] == maxvol or voltc[i][events[i][b]] >= maxvol):
                    p_height[i] += [maxvol]
                else:
                    p_height[i] += [voltc[i][events[i][b]]]
            
        d = str(datetime.fromtimestamp(int(self.filepath[-18:-8])).date())
        if d in ['2024-04-23', '2024-04-25']:
            return events, p_height, mw, [0] * len(events)
        
        return events, p_height, mw, self.run_number()
    
#interesting files: '/eos/experiment/gbar/pgunpc/data/24_05_16/24_05_16lya/LY1234.1715884249.817.trc', '/eos/experiment/gbar/pgunpc/data/24_05_23/24_05_23lya/LY1234.1716493289.096.trc' 0.2906665578484535
#look at: '/eos/experiment/gbar/pgunpc/data/24_05_23/24_05_23lya/LY1234.1716489069.381.trc' if peak of around height 0.12 is available

