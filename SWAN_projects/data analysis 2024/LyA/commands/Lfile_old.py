#!/usr/bin/env python
# coding: utf-8

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
from tqdm import tqdm


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


def parfunc(x,a,c,d):
    '''
    
    parabola which we can fit to the pulses
    
    Parameters
    ------------
    x: variable of the function
    a: factor in front of the quadratic term
    c: horizontal offset of the parabola
    d: vertical offset of the parabola
    
    Returns
    ------------
    value of the parabola given by the parameters at point x
    
    '''
    return a*(x-c)**2 + d #a*(x-c)**2 + b*(x-c) + d


def gaussfunc(s):
    '''
    
    get heights at real numbers for a gaussian with sigma = s
    
    Parameters
    ------------
    s: sigma of the gaussian
    
    Returns
    ------------
    array with the values of the gaussian at the whole numbers from -4s to +4s
    
    '''
    gauss = lambda x: 1/(s*(2*np.pi)**0.5) * np.exp(-1/2 * x**2/s**2)
    
    xl = np.linspace(-4*s, 4*s, 8*s+1)
    
    return gauss(xl)


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
    #f = w/(2.*np.pi)
    fitfunc = lambda x: A * np.sin(w*x + p) + c
    return fitfunc, A, w


def parfit(xx, yy, p = [2e+15,0,-0.05]):
    '''
    
    get a parabola fit for the points around a peak
    
    Parameters
    ------------
    xx: x coordinates of the data
    yy: y coordinates of the data
    
    Returns
    ------------
    fitfunc: fit parabola to xx and yy
    
    '''
    xx = np.array(xx)
    yy = np.array(yy)
    popt, pcov = scio.curve_fit(parfunc, xx, yy, p0 = p, maxfev = 2000)
    fit = lambda x: popt[0]*(x-popt[1])**2 + popt[2]
    return fit


def fitgaussfunc(x,A,s,m):
    '''
    
    get heights at real numbers for a gaussian with sigma = s
    
    Parameters
    ------------
    s: sigma of the gaussian
    
    Returns
    ------------
    array with the values of the gaussian at the whole numbers from -4s to +4s
    
    '''
    return A/(s*(2*np.pi)**0.5) * np.exp(-1/2 * (x-m)**2/s**2)# + b


def gaussfit(xx, yy, p):
    '''
    
    get a parabola fit for the points around a peak
    
    Parameters
    ------------
    xx: x coordinates of the data
    yy: y coordinates of the data
    
    Returns
    ------------
    fitfunc: fit parabola to xx and yy
    
    '''
    xx = np.array(xx)
    yy = np.array(yy)
    popt, pcov = scio.curve_fit(fitgaussfunc, xx, yy, p0 = p)
    return popt


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
    
    
    def ts_from_ms(self, ms):
        time = self.read_trc()[0][0]
        #tott = time[-1]-time[0]
        dt = time[1]-time[0]
        #steps = len(time)-1
        return round((ms-time[0])/dt, 1)
    
    
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
        
        
    def parpeak(self, ch, ts):
        '''

        get the actual peak and timing for a voltage pulse from the LyA trc files by fitting a parabola to the 3 elements before the highest measured voltage and 1 point after
        if the parabola has a loweer peak than the voltage point, we keep the voltage as the pulse height

        Parameters
        ------------
        ch: channel in which the peak was recorded
        ts: timestep of the peak

        Returns
        ------------
        t: time of the peak of the parabola
        h: height of the peak of the parabola or max measured pulse voltage, whichever is higher (gives the peak as a positive number)

        '''
        data = self.read_trc()
        time = data[0][0]
        
        av = np.average(data[1][ch][8000:10000])
        
        #dt = time[1]-time[0] #time between two measurement points
        xc = time[ts-3:ts+2]
        yc = [val-av for val in data[1][ch][ts-3:ts+2]]
        xl = np.linspace(xc[0], xc[-1], 100)
        
        try:
            fit = parfit(xc, yc, p = [2e+15, xc[3], yc[3]])
        except:
            t, h = ts, data[1][ch][ts]
            fit = lambda x: 0
        else:
            m = np.argmin(fit(xl))
            t, h = xl[m], min(fit(xl[m]), data[1][ch][ts]-av)
        


        return t, h
    
    
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
        fre = [0,0,0,0] #for the term in front of x in the sinus

        for i in range(4):
            try:
                ret[i] = sinfit(xx[8000:10000], yy[i][8000:10000])
            except:
                func[i] = lambda x: 0
                amp[i] = 0
                fre[i] = 0
            else:
                func[i] = ret[i][0]
                amp[i] = ret[i][1]
                fre[i] = ret[i][2]

        return func, amp, fre
    
    
    #plot the data from the 4 channels with the peaks
    def plot_voltage(self,p = 0.001, h = 0.005, d = 10, back = 30, trange = None, comb = False, no_cor = False, show_fit = False, tof = False, gauss = False, ave = 0, ch = 0, save = False, test = False):
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
        gauss: default False, if gauss is an int, then smooth the voltages with a gaussian distribution which has sigma = s
        ave: default 0, if not 0 then we take for each voltage the average voltage, going left and right ave steps each
        ch: default 0, if ch is 0 we get one figure for each channel, if ch is 1,2,3 or 4, we only get that channel
        save: if save is not False, the figure gets saved at /eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA MCPs images/, the name of the file is whatever the string of save is

        '''
        if trange == None: cut_low, cut_high = 0,10001
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
        
        if type(gauss) == int:
            garr = gaussfunc(gauss)
            gvolt = [0,0,0,0] #put the 4 gauss smoothed voltage array in here if gauss is an int
            
        if ave != 0: 
            aconv = [1/(2*ave+1)] * (2*ave+1)
            avolt = [0,0,0,0]
        
        #fit a sin function to the voltage of the 4 channels and subtract it from the voltage if mw is on
        if mw == 1:
            xl = np.linspace(0, len(time), len(time)+1)
            fitc, ampc, frec = self.fit_file() #list with the sinus fits for the channels
            for i in range(4):
                #only apply the sinfit if the amplitude of the sin is bigger than 0.0015V
                if abs(ampc[i]) >= 0.0015 and 0.2 < frec[i] < 0.5:
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
            fig = plt.figure(figsize = (30,10))
            
            for i in range(4):
                chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
                plt.scatter(time[cut_low:cut_high], voltc[i][cut_low:cut_high], label = 'Voltage channel ' + str(i))
                plt.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 100)
                
                if type(gauss) == int:
                    gvolt[ch] = np.convolve(voltc[ch], garr)[4*gauss:-4*gauss]
                    #plt.scatter(time[cut_low:cut_high], gvolt[ch][cut_low:cut_high])
                    
            if type(gauss) == int: plt.scatter(time[cut_low:cut_high], [a+b+c+d for a,b,c,d in zip(gvolt[0],gvolt[1],gvolt[2],gvolt[3])], label = 'combined gauss smoothed voltage')
            plt.legend(loc = 'best')
            plt.xlabel(xlabel = 'Time [µs]')
            plt.ylabel(ylabel = 'Voltage [V]')
            plt.title(label = 'Voltages and peaks of the four channels')
            return
        
        if ch != 0:
            plt.figure(figsize = (30,10))
            plt.grid(True, alpha = 0.7)
            ch = ch-1
            chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[ch]],height_peaks[ch]) if time[cut_low] <= val <= time[cut_high - 1]]
            plt.scatter(time[cut_low:cut_high], voltc[ch][cut_low:cut_high], s = 15, label = 'Corrected voltage', color = '#1f77b4')
            plt.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 30, linewidth = 3, color = '#ff7f0e')
            
            if no_cor:
                plt.scatter(time[cut_low:cut_high], volt[ch][cut_low:cut_high], s = 15, label = 'Uncorrected voltage', color = '#7f7f7f')
                plt.legend(loc = 'upper right')
                
            if show_fit:
                if mw == 1:
                    if abs(ampc[ch]) >= 0.0015 and 0.2 < frec[ch] < 0.5:
                        plt.plot(np.linspace(time[cut_low],time[cut_high], 2* (cut_high - cut_low)), fitc[ch](np.linspace(cut_low,cut_high, 2*(cut_high-cut_low))), color = '#d62728', label = 'Sinus fit')
                    else:
                        plt.plot([time[cut_low], time[cut_high]], [av[ch], av[ch]], color = '#d62728', label = 'Average')
                else: plt.plot([time[cut_low], time[cut_high]], [av[ch], av[ch]], color = '#d62728', label = 'Average')
                plt.legend(loc = 'upper right')
                
            if tof == True:
                plt.axvline(window[0], color = 'black', linestyle = '--')
                plt.axvline(window[1], color = 'black', linestyle = '--')
                
            if type(gauss) == int:
                gvolt[ch] = np.convolve(voltc[ch], garr)[4*gauss:-4*gauss]
                plt.scatter(time[cut_low:cut_high], gvolt[ch][cut_low:cut_high], label = 'gauss smoothed voltage', color = '#2ca02c')
                plt.legend(loc = 'upper right')
                
            if ave != 0:
                avolt[ch] = np.convolve(voltc[ch], aconv)[ave:-ave]
                plt.scatter(time[cut_low:cut_high], avolt[ch][cut_low:cut_high], label = 'averagely smoothed voltage', color = '#9467bd')
                
            if test == True:
                for k in range(len(chpeaks)):
                    ystretch = (chpeaks[k][1] + max(voltc[i][self.ts_from_ms(chpeaks[k][0])-30:self.ts_from_ms(chpeaks[k][0])+30])) / (ytest[30]-ytest[22])
                    ycurr = [val * ystretch for val in ytest]
                    yoffs = ycurr[30] - chpeaks[k][1]
                    ycurr = [-(val - yoffs) for val in ycurr]
                    plt.scatter([val + chpeaks[k][0] for val in xtest], ycurr, color = 'red', s = 10)
                    #plt.scatter([val + chpeaks[k][0] for val in xtest][10:50], [-val * chpeaks[k][1]/ytest[30] for val in ytest][10:50], color = 'green', s = 10)
                    #plt.scatter([val + chpeaks[k][0] for val in xtest][20:40], [-val * chpeaks[k][1]/ytest[30] for val in ytest][20:40], color = 'orange', s = 10)
                
            plt.ylabel(ylabel = 'Voltage [V]')
            plt.xlabel(xlabel = 'Time [s]')#'Time [µs]')
            plt.title(label = 'Voltage and peaks in channel %i' % (ch+1))
            
            return
        
        fig = plt.figure(layout = 'tight', figsize = (30,12))
        gs = GridSpec(2, 2, figure = fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        
        for i, ax in enumerate(fig.axes):
            chpeaks = [[val,bal] for val,bal in zip(time[time_peaks[i]],height_peaks[i]) if time[cut_low] <= val <= time[cut_high - 1]]
            #ax.grid(True, alpha = 1)
            ax.scatter(time[cut_low:cut_high], voltc[i][cut_low:cut_high], s = 15, label = 'Corrected voltage', color = '#1f77b4')
            ax.scatter([val[0] for val in chpeaks], [-val[1] for val in chpeaks], marker = 'x', s = 30, linewidth = 3, color = '#ff7f0e')
            
            if no_cor:
                ax.scatter(time[cut_low:cut_high], volt[i][cut_low:cut_high], s = 15, label = 'Uncorrected voltage', color = '#7f7f7f')
                ax.legend(loc = 'upper right')
                
            if show_fit:
                if mw == 1: 
                    if abs(ampc[i]) >= 0.0015 and 0.2 < frec[i] < 0.5:
                        ax.plot(np.linspace(time[cut_low],time[cut_high], 2* (cut_high - cut_low)), fitc[i](np.linspace(cut_low,cut_high, 2*(cut_high-cut_low))), color = '#d62728', label = 'Sinus fit')
                    else:
                        ax.plot([time[cut_low], time[cut_high]], [av[i], av[i]], color = '#d62728', label = 'Average')
                else: ax.plot([time[cut_low], time[cut_high]], [av[i], av[i]], color = '#d62728', label = 'Average')
                ax.legend(loc = 'upper right')
                
            if tof == True:
                ax.axvline(window[0], color = 'black', linestyle = '--')
                ax.axvline(window[1], color = 'black', linestyle = '--')
                
            if type(gauss) == int:
                gvolt[i] = np.convolve(voltc[i], garr)[4*gauss:-4*gauss]
                ax.scatter(time[cut_low:cut_high], gvolt[i][cut_low:cut_high], label = 'gauss smoothed voltage', color = '#2ca02c')
                ax.legend(loc = 'upper right')
                
            if ave != 0:
                avolt[i] = np.convolve(voltc[i], aconv)[ave:-ave]
                plt.scatter(time[cut_low:cut_high], avolt[i][cut_low:cut_high], label = 'averagely smoothed voltage', color = '#9467bd')
                
            if test == True:
                for k in range(len(chpeaks)):
                    ystretch = (chpeaks[k][1] + max(voltc[i][self.ts_from_ms(chpeaks[k][0])-30:self.ts_from_ms(chpeaks[k][0])+30])) / (ytest[30]-ytest[22])
                    ycurr = [val * ystretch for val in ytest]
                    yoffs = ycurr[30] - chpeaks[k][1]
                    ycurr = [-(val - yoffs) for val in ycurr]
                    ax.scatter([val + chpeaks[k][0] for val in xtest], ycurr, color = 'red', s = 10)
                    #ax.scatter([val + chpeaks[k][0] for val in xtest][10:50], [-val * chpeaks[k][1]/ytest[30] for val in ytest][10:50], color = 'green', s = 10)
                    #ax.scatter([val + chpeaks[k][0] for val in xtest][20:40], [-val * chpeaks[k][1]/ytest[30] for val in ytest][20:40], color = 'orange', s = 10)
                
            ax.set_ylabel(ylabel = 'Voltage [V]')
            ax.set_xlabel(xlabel = 'Time [s]')#'Time [µs]')
            ax.set_title(label = 'Voltage and peaks in channel %i' % (i + 1))
        
        if save != False:
            fig.patch.set_alpha(1.0)
            fig.savefig('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data analysis/Figures/LyA MCPs images/' + str(save), dpi = 400)
        
        return #time_peaks, height_peaks
    
    
    #get events from one file, return list with 4 lists where we can see where in the data of the 4 seperate channels there are peaks and a list with the peak heights
    #mw will return 0 (if the mw is turned off) or 1 (if the mw is turned on)
    def get_events_from_file(self, prom = 0.001, hgt = 0.005, dist = 10, l = 30, trange = None, ch = -1, pc = -1, pr = False, rv = False):
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
            fitc, ampc, frec = self.fit_file() #list with the sinus fits for the channels
            for i in range(4):
                #only apply the sinfit if the amplitude of the sin is bigger than 0.0015V
                if abs(ampc[i]) > 0.0015 and 0.2 < frec[i] < 0.5:
                    voltc[i] = [val + fitc[i](bal) for val,bal in zip(volt[i],xl)] #plus because volt is the negative of the recorded voltage
                else:
                    av = np.average(volt[i][8000:10000])
                    voltc[i] = [val - av for val in volt[i]]
                     
        #if mw is off, we subtract the average of voltc[8000:10000] from voltc
        else:
            for i in range(4):
                av = np.average(volt[i][8000:10000])
                voltc[i] = [val - av for val in volt[i]]
        
        #maxvol changes depending on the LyA voltage, currently just change by looking at the date, later implement this better
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
            events[i] = peaks[0] #set the ith element of the list 'events' to the locations of the peaks
            events[i] = [val for val in events[i] if val > cut_low and val < cut_high] #only take peaks in trange
            
            ###################
            cch, ph, pr = self.args()
            ###################

            
            #go through all the potential peaks
            for m in range(len(events[i])):
                #we want to disregard peaks with certain features (we set events[i][m] = -1 for the ones we want to discard) (these criteria have been found through trial and error, i am sure that there are better ones)
                currh = voltc[i][events[i][m]] #height of the current peak
                currt = events[i][m] #timestep of the current peak
                
                if (l > 10):
                    #look at the voltage elements between 10 and l (default 25) positions before the peak and calculate the standard deviation
                    tl = voltc[i][(currt-l) : (currt-10)] #list with voltage for the datapoints from l before the mth peak of the ith channel until 10 before the peak
                    if len(tl) == 0: tl = [0,1] #if the length of tl is 0, we define it as [0,1]
                    mean = sum(tl) / len(tl) #mean of the elements in tl
                    variance = sum([((x - mean) ** 2) for x in tl]) / len(tl) #variance of the elements in tl
                    std = variance ** 0.5 #standard deviation of the elements in tl
                    
                    tln = voltc[i][(currt+15) : (currt+l)] #list with voltage for the datapoints from 10 after the mth peak of the ith channel until 30 after the peak
                    if len(tln) == 0: tln = [0,1] #if the length of tln is 0, we define it as [0,1]
                    meann = sum(tln) / len(tln) #mean of the elements in tln
                    variancen = sum([((x - meann) ** 2) for x in tln]) / len(tln) #variance of the elements in tln
                    stdn = variancen ** 0.5 #standard deviation of the elements in tln
                    
                    #get the average value of the 5 voltages before the peak
                    av = sum([val for val in voltc[i][(events[i][m]-5):events[i][m]]])/5
                    
                    #if currt is in the first 2000 elements, we disregard the peak
                    if currt < 2000:
                        events[i][m] = -1
                        currh = 10
                    
                    #we keep all peaks with greater height than 0.05V
                    if currh <= 0.05:
                        
                        if currt <= 3500 and currh <= 0.02:
                            events[i][m] = -1
 
                            
                        left = 3
                        if voltc[i][currt-1] == currh: left = 4
                        
                        if -2205 < float(self.df[self.df.LyA == self.filepath]['Ly_MCP_1']) < -2195:
                            backv = 200
                        else:
                            backv = 200
                        
                        #check if the difference betwen the peak voltage and 5 elements before the peak is greater than 0.02 or the standard deviation divided by peak height is larger than 0.25
                        if currh - abs(voltc[i][currt-5]) < 0.04 and currh - voltc[i][currt-5] < 0.06 and (currh - voltc[i][currt-left] < currh*0.9 or (std+stdn)/currh > 0.6):

                            #if the standard deviation divided by the peak height is bigger than 0.15 and the surrounding elements do not fulfill certain conditions we discard the peak 
                            '''try here also the difference if we have 0*std/currh instead of 3*std/currh and if the last criteria is good or bad'''
                            if (events[i][m] != -1 and std/currh > 0.13 and 
                                 (max(voltc[i][currt-20:currt-5]) >= currh/(2+3*std/currh) or min(voltc[i][currt:currt+20]) <= -currh/1.9 or max(voltc[i][currt+10:currt+25]) >= currh/1.5)
                               and max(voltc[i][currt-backv:currt]) >= 2*currh):

                                events[i][m] = -1

                            if events[i][m] != -1 and av < 0.002 and currh < 0.04:

                                events[i][m] = -1
                                
                        #if the peak is smaller than 0.0085V the standard deviation before and after the peak has to be smaller than 0.001 and there has to be no voltage >= currh/3 in the 20 voltages before and after to keep it
                        if events[i][m] != -1 and currh < 0.008:
                            if (std > 0.001 or stdn > 0.001) and max([abs(val) for val in voltc[i][currt-30:currt-4]+voltc[i][currt+5:currt+20]]) >= currh/3:

                                events[i][m] = -1
                                
                        #if the peak is smaller than the voltage 4 elements before the peak plus 1/5 of the peak, discard it
                        if events[i][m] != -1 and currh <= voltc[i][currt-4] + currh/5:

                            events[i][m] = -1
                            
                        #if the peak is smaller than the voltage 6 elements after the peak plus 1/5 of the peak, discard it
                        if events[i][m] != -1 and currh <= voltc[i][currt+6] + currh/5:

                            events[i][m] = -1
                        
                        #if there is a higher voltage than 0.08V in the 30 elements before the peak and peak is smaller than 0.04V, discard it

                            
                        #if there is a higher voltage than 0.08V in the 200 elements before the peak and peak is smaller than 0.02V, discard it
                        'maybe should be 0.02 or 0.03 instead of 0.015'
                        if (events[i][m] != -1 and max(voltc[i][currt-200:currt]) >= 0.08 and currh <= 0.02 and (std/currh > 0.15 or (stdn + std)/currh > 0.25) and currh - voltc[i][currt-left] < 0.9*currh): 

                            events[i][m] = -1
                            
                        if (events[i][m] != -1 and max(voltc[i][currt-200:currt]) >= 0.08 and currh <= 0.04 and std/currh > 0.15 and currh < 2*max(voltc[i][currt-20:currt-4] + voltc[i][currt+5:currt+20])): 

                            events[i][m] = -1
                            
                        #if there is a voltage element in the 15 points before the peak that is negative and its absolute is bigger than 0.5 of the peak height and there is a voltage greater than 2*currh in the last 200 elements, discard the peak
                        if abs(min(voltc[i][currt-15:currt])) >= currh/2 and max(voltc[i][currt-200:currt]) >= 2*currh:

                            events[i][m] = -1
                            
                        #if std/currh > 0.4 and the max of the last 30 elements is bigger than 1.2*currh, discard the peak
                        if std/currh > 0.4 and (stdn/currh > 0.15 or max(voltc[i][currt-15:currt]) >= 2*currh) and max(voltc[i][currt-30:currt]) > 1.2*currh:

                            events[i][m] = -1
                            

                            
                       
                    
                    #if there is a peak that is bigger than maxvol/2 in the last 30 voltages, discard any peak smaller than maxvol/3.5
                    if (events[i][m] != -1 and max(voltc[i][currt-30:currt]) >= maxvol/2 and currh <= maxvol/3.5):

                        events[i][m] = -1
                        
                    #if there is a peak that is bigger than maxvol/3 in the last 20 voltages, discard any peak smaller than maxvol/6
                    if (events[i][m] != -1 and max(voltc[i][currt-20:currt]) >= maxvol/3 and currh <= maxvol/6):

                        events[i][m] = -1
                    #if i == ch and abs(currt-pc) <= 1:
                        #return [round(val,20) for val in time[currt-50:currt+50]], [round(val,10) for val in voltc[i][currt-50:currt+50]]

                    

                    
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
    
    
    def corr_func():
        '''

        the shape of the voltage pulse from 10 points before the peak to 1000 points after is taken. This is done for 4 files, each having only 1 pulse at around 0.1 V. We use the Lfile.parpeak function to get the 'actual'
        time and height of each pulse. The times are then shifted such that the actual pulse timing is at 0. The voltages are all normalized so that the actual pulse height is 0.1V. Then the datapoints from the 4 pulses are
        put together into an array. With this array we define a function, which first searches for the point in the combined array just before the input x. Then a polynomial of second degree is fit for the 9 points around
        the one we found (4 before to 4 after). The returned value is the value of the polynomial at x. We evaluate this function at 10000 points (from the the first timestep in the combined array until the last). This
        process is repeated for each channel.

        The arrays for each channel, together with the corresponding times, are all put into one array of shape (2, 4). In the first subarray are the times for each channel and in the second one are the voltages. This array
        is finally saved at '/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy'.

        These values can be used to eliminate the noise which comes with each pulse from the LyA trc files.

        I still need to check if the shape stays the same for other MCP voltages (all these are for 2.2 kV).
        
        The files which are used to get the corrections are:
        
        #ch0 of these files were used to get ch0 correction:
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001663.856.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001891.844.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001906.306.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001948.859.trc'
        
        #ch1 of these files were used to get ch1 correction:
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001763.802.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717002278.882.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717002522.342.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717008930.758.trc'
        
        #ch2 of these files were used to get ch2 correction:
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001663.856.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717002037.857.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717003087.669.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717003199.268.trc'
        
        #ch3 of these files were used to get ch3 correction:
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717001735.319.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717002094.854.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717002564.874.trc'
        #'/eos/experiment/gbar/pgunpc/data/24_05_29/24_05_29lya/LY1234.1717003329.927.trc'

        '''
        a = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy') #get the array with all the correction arrays
        
        a[1] = [bal for val,bal in sorted(zip(a[0],a[1]))][1:] #sort the voltages with time
        a[0] = sorted(a[0])[1:] #sort time
        
        at = -1.9027936307608238e-10 #time of the peak of a we get from fitting a parabola from ap.argmax(y)-4:np.argmax(y)+3
        ah = 0.20067667063851166 #height of the peak of a we get from fitting a parabola from ap.argmax(y)-4:np.argmax(y)+3
        
        a[0] = [val+at for val in a[0]] #adjust the timing of a so it has the peak at 0
        a[1] = [val*(0.2/ah) for val in a[1]] #adjust the height of a so it has peak height 0.2

        return a
    
    
    def args(self):
        return -1, -1, False
