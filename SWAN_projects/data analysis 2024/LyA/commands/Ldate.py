#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/Data loader')
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime
import scipy.signal as sci
from Lfile import Lfile
import fnmatch, re
from tqdm import tqdm
from Ltif import Lpicday
from gbarDataLoader24 import loadShortSummary
import time


# In[7]:


#Define class, which is a date on which data with the Lyman alpha detector was taken (everything will be searched for in '/eos/experiment/gbar/pgunpc/data/'), date should be a string written like year_month_day (each of them 2 characters)
class Ldate:
    '''
    
    Class to help organize and get data from the Lyman alpha MPCs for a specific date. The main functions are get_events_from_day, add_date, remove_date and read.
    
    Parameters
    ------------
    date in the format yy_mm_dd
    
    '''
    
    #define instance method
    def __init__(self, date, corr_df = None):
        self.date = date
        if type(corr_df) != type(None):
            self.corr_df = corr_df.fillna(-100)
        else:
            self.corr_df = pd.read_csv('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/correction_arrays.csv').fillna(-100)
        
        
    #define string
    def __str__(self):
        return f'{self.date}'
    
    
    #check if a date has data for the Lyman alpha experiment
    def check_date(self):
        '''
        make sure that there is a lya data for the date
        
        Returns
        ------------
        True or False
        
        '''
        datapath = Path('/eos/experiment/gbar/pgunpc/data/' + self.date) #Path to folder for date
        return datapath.is_dir()
    
    
    #get list with all the filenames for the date
    def get_filelist(self):
        '''
        get a list with the names of the trc files containing the data collected by the 4 channels in the lya setup
        
        Returns
        ------------
        filelist = list with trc filenames
        
        '''
        datapath = Path('/eos/experiment/gbar/pgunpc/data/' + self.date + '/' + self.date + 'lya') #make datapath into a path object
        
        #test if this folder exists, if it does not then there are no measurements available from the Lyman alpha detectors for the input date
        if (self.check_date() == False):
            return print('There are no Lyman alpha measurements available for the date', self.date + '.')

        filelist = [val for val in os.listdir(datapath) if val.endswith('.trc') and val.startswith('LY1234')] #only take filenames that start wich LY1234
        
        return filelist
    
    
    #get list with all the filespaths for the date
    def get_filepaths(self):
        '''
        get a list with the paths to the lya trc files
        
        Returns
        ------------
        filepaths = list with paths to the trc files for the date
        
        '''
        filenames = self.get_filelist() #get the names of the files
        
        filepaths = ['/eos/experiment/gbar/pgunpc/data/' + self.date + '/' + self.date + 'lya/' + val for val in filenames] #list with the paths to the files, return this in the end
        
        return filepaths
    
    
    #get times of measurements in filelist
    def __measurement_time(filelist):
        '''
        get the times at which the files were generated
        
        Parameters
        ------------
        filelist = names of files of which we want the measurement times
        
        Returns
        ------------
        time = list with times for the files in the format datetime
        
        '''
        unixts = [int(val[-18:-8]) for val in filelist] #list with unix timestamps from the filenames, change them to integers
        
        time = [datetime.fromtimestamp(val) for val in unixts] #list with actual times, which is what we return with this function
        
        return time
    
    
    #get a plot for the measurement times for the date
    def plot_measurement_time(self):
        '''
        plot at what times of the day the measurements were taken (both tif and trc files)
        
        '''
        a = Ldate(self.date)
        
        tif = Lpicday(self.date).get_filelist() #tif filelist
        trc = a.get_filelist() #trc filelist
        
        time_trc = Ldate.__measurement_time(trc) #get list with the measurement times for the trc files
        time_tif = Ldate.__measurement_time(tif) #get list with the measurement times for the tif files
        
        plt.figure(figsize=(10,10)) #start a new figure
        
        ax = plt.gca() #define ax as the axis of the plot
        xfmt = mdates.DateFormatter('%d.%m %H:%M') #prepare the format for the x axis (they are now dat.month hour:minute)
        ax.xaxis.set_major_formatter(xfmt) #set the x axis format as the one defined above
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        y_value1 = [1.0] * len(time_trc) #set an arbitrary value on the y axis for the trc files
        y_value2 = [0.5] * len(time_tif) #set an arbitrary value on the y axis for the tif files
        
        plt.scatter(time_trc, y_value1, label = 'trc file times', color = 'darkorange') #make the scatter plot
        plt.scatter(time_tif, y_value2, label = 'tif file times', color = 'lime')
        plt.title('Lyman alpha detector measuring times on ' + self.date)
        plt.xticks(rotation = 45) #rotate the labels on the x axis by 45 degrees
        plt.legend(loc='best')
        plt.ylim([0,2]) #limit the y-axis
        plt.show()
        
        return
    
    
    #get the time and height of peaks from a date for each detector and their filepath
    #prom, height and distance refer to parameters used to determine what counts as a peak in get_events_from_file
    def get_events_from_day(self, prom = 0.001, hgt = 0.005, dist = 10, back = 30, trange = None):
        '''
        get information about the events recorded by the 4 mcps in the lya setup for the date
        
        Parameters
        ------------
        prom = 0.001 is a parameter used to determine the peaks
        hgt = 0.005 is the minimum height for a peak to be recorded
        dist = 10 the distance two peaks have to be apart from eachother
        back = 30 how many datapoints we go back to determine the std, with which we determine whether something is a peak or not
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        
        Returns
        ------------
        all the returns have 4 lists, 1 list for each channel. Events at the same index and channel correspond to the same event.
        
        time = times of the events
        height = heights of the events
        loc = filepath for the event
        mw = 0 if the microwave was turned off and 1 if it was on for an event
        
        '''
        if trange == None: cut_low, cut_high = 0, 10002
        else: cutoff_low, cutoff_high = trange[0], trange[1]
        
        #check if the date has data
        if (self.check_date == False):
            return print('There are no Lyman alpha measurements available for the date', self.date + '.')
        
        filepaths = self.get_filepaths() #list of the paths to the files
        
        filenums = len(filepaths) #how many files there are for the date
        
        df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t') #datafile to give to Lfile
        
        #events = [0] * 4 #list for the numbers of events detected by each detector
        
        time = [] #list to put in the times of the peaks for the 4 detectors 
        height = [] #list to put in the heights of the peaks for the 4 detectors 
        loc =  [] #list to put the filepath for the peaks
        mw =  [] #list to put whether the mw was turned on
        run = [] #list to put the run number in
        #we need to remove the first elements of these in the end
        
        #go through the files
        for i in tqdm(range(filenums)):
            #define Lfile object
            file = Lfile(filepaths[i], df = df, corr_df = self.corr_df)
            
            filedata = file.get_events_from_file(hgt = hgt, dist = dist, trange = trange) #get the time and height of the events for the channels for the ith file in filepaths   
            loc += [filepaths[i]]  #add the location of the current file to loc
            mw += [filedata[2]]  #add list with info if mw is turned on (1) or off (0)
            time += [list(filedata[0])] #add the times of the peaks of the current file to times
            height += [list(filedata[1])] #add the heights of the peaks of the current file to heights
            run += [filedata[3]]
               
            
        return time, height, loc, mw, run
    
    
    def peaks_data(self):
        '''
        
        Create a file at /eos/user/l/lkoller/GBAR/data24/datasummary24/yy_mm_dd which incldes the data for the peaks found for the date. 
        In the first line the microwave status (either on or off) is saved, in the second one the corresponding channel, then the file location from which the peak is and finally the time and height of the peak.
        
        Returns
        ------------
        df = the dataframe from the file that is created, if there is not data in one of the elements (in the times and heights columns) a 'NaN' is inserted. The columns are:
            LyA: filepath to the current trc file from the Lyman alpha MCPs
            run: run number of the current event (0 if the file is not part of any run)
            microwave: the microwave status, either on or off
            NE50_I: number of particles in millions in the NE50 line (so how many particles we get from elena)
            time_ch1: for each file a list with the timesteps of the peaks found in channel 1
            height_ch1: for each file a list with the heights in Volt of the peaks found in channel 1
            time_ch2: for each file a list with the timesteps of the peaks found in channel 2
            height_ch2: for each file a list with the heights in Volt of the peaks found in channel 2
            time_ch3: for each file a list with the timesteps of the peaks found in channel 3
            height_ch3: for each file a list with the heights in Volt of the peaks found in channel 3
            time_ch4: for each file a list with the timesteps of the peaks found in channel 4
            height_ch4: for each file a list with the heights in Volt of the peaks found in channel 4
            
        '''
        #make sure that there are files to use for the current date
        if len(self.get_filepaths()) == 0:
            return print('There are no LyA trc file for ' + self.date + '.')
        
        LYAFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date / 'LyA_data'
        
        if ((LYAFILE / ('peaks_' + self.date + '.npy')).is_file() == True):
            print('The peaks datafile for the date ' + self.date + ' already exists.')
            return self.read()
        
        #create a folder for the current date if it does not already exist
        if not (Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date).exists():
            Path.mkdir(Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date)
        
        #create the LyA_data folder if it does not already exist
        if not (LYAFILE).exists():
            Path.mkdir(LYAFILE)
            
        #columns of the file
        cols = ['LyA', 'run', 'microwave', 'time_ch1', 'height_ch1', 'time_ch2', 'height_ch2', 'time_ch3', 'height_ch3', 'time_ch4', 'height_ch4']
            
        pdata = self.get_events_from_day() #get the data of the peaks that we find
        mw = ['off' if val == 0 else 'on' for val in pdata[3]] #list with mw on/off for each peak
        run = pdata[4]
        file = pdata[2] #list with the file path for each peak
        
        #get the lists with the times and heights of the peaks for each channels
        time1 = [val[0] if len(val[0]) > 0 else 'NaN' for val in pdata[0]] #list with the time of each peak for channel 1
        height1 = [val[0] if len(val[0]) > 0 else 'NaN' for val in pdata[1]] #list with the height of each peak for channel 1
        time2 = [val[1] if len(val[1]) > 0 else 'NaN' for val in pdata[0]] #list with the time of each peak for channel 2
        height2 = [val[1] if len(val[1]) > 0 else 'NaN' for val in pdata[1]] #list with the height of each peak for channel 2
        time3 = [val[2] if len(val[2]) > 0 else 'NaN' for val in pdata[0]] #list with the time of each peak for channel 3
        height3 = [val[2] if len(val[2]) > 0 else 'NaN' for val in pdata[1]] #list with the height of each peak for channel 3
        time4 = [val[3] if len(val[3]) > 0 else 'NaN' for val in pdata[0]] #list with the time of each peak for channel 4
        height4 = [val[3] if len(val[3]) > 0 else 'NaN' for val in pdata[1]] #list with the height of each peak for channel 4
        
        df = pd.DataFrame([file, run, mw, time1, height1, time2, height2, time3, height3, time4, height4], index = cols) #dataframe which we want to save
        df = df.transpose()
        
        df = df.fillna('None')
        df = df.replace('NaN', 'None')
        
        np.save(LYAFILE / ('peaks_' + self.date), df) #use np.save to recover the correct type of the data when reading it (self.read() has to be used)
        
        return self.read()
    
    
    def read(self):
        '''
        
        read the csv file and get a pandas dataframe with the correct types for the columns
        
        '''
        LYAFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date / 'LyA_data'
        
        if not (LYAFILE / ('peaks_' + self.date + '.npy')).exists():
            print('There is no peaks data file for the date ' + self.date + '.')
            return
        
        if (self.date == '24_04_23' or self.date == '24_04_25'):
            return pd.DataFrame(np.load(LYAFILE / ('peaks_' + self.date + '.npy'), allow_pickle = True), columns = ['LyA', 'microwave', 'time ch1', 'height ch1', 'time ch2', 'height ch2', 'time ch3', 'height ch3', 'time ch4', 'height ch4'])
        
        #columns of the file
        cols = ['LyA', 'run', 'microwave', 'time_ch1', 'height_ch1', 'time_ch2', 'height_ch2', 'time_ch3', 'height_ch3', 'time_ch4', 'height_ch4']
        
        df = pd.DataFrame(np.load(LYAFILE / ('peaks_' + self.date + '.npy'), allow_pickle = True), columns = cols)
        
        return df
    
    
    def read_old(self):
        '''
        
        read the csv file (old one) and get a pandas dataframe with the correct types for the columns
        
        '''
        LYAFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' / self.date / 'LyA_data'
        
        if not (LYAFILE / ('peaks_' + self.date + '.npy')).exists():
            print('There is no peaks data file for the date ' + self.date + '.')
            return
        
        #columns of the file
        cols = ['LyA', 'run', 'microwave', 'time_ch1', 'height_ch1', 'time_ch2', 'height_ch2', 'time_ch3', 'height_ch3', 'time_ch4', 'height_ch4']
        
        if (self.date == '24_04_23' or self.date == '24_04_25'):
            return pd.DataFrame(np.load(LYAFILE / ('peaks_' + self.date + '.npy'), allow_pickle = True), columns = cols)
            
        df = pd.DataFrame(np.load(LYAFILE / ('peaks_' + self.date + '_old.npy'), allow_pickle = True), columns = cols)        
        
        return df
        