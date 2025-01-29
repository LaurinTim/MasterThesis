#!/usr/bin/env python
# coding: utf-8

# In[64]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')
sys.path.append('/eos/home-i00/l/lkoller/SWAN_projects/commands/data_loader')

from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from Ldate import Ldate
from Ltif import Lpicday
from Waveform import wf
from gbarDataLoader24 import loadShortSummary, loadDatafile, loadElena
from ast import literal_eval
import time
pd.set_option("display.max_rows",90)
pd.set_option("display.max_columns",None)


# In[116]:


#path to the file
LYADFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24'
LYASAVE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24'

#columns of the file
cols = ['Date', 'Time', 'LyA', 'MCP5_picture', 'MCP5_Waveform', 'run', 'microwave', 'NE50_I', 'time_ch1', 'height_ch1', 
        'time_ch2', 'height_ch2', 'time_ch3', 'height_ch3', 'time_ch4', 'height_ch4', 'MCP5_volt_sum', 'beam_start', 'beam_stop']

    
def LyA_removeDate(date):
    '''

    Remove the data from self.date from LyAdata24.npy
    
    Parameters
    ------------
    date: date which we want to remove in the format yy_mm_dd

    '''              
    #if LyApeaks24.txt does not exist, we just stop
    if (LYADFILE / 'LyAdata24.npy').is_file() == False:
        return print('The LyAdata24 file does not exist yet.')

    #get the datafile
    datafile = read_df()
    datafile = datafile.set_index(keys = 'Date')
    
    #check if the date is in the datafile
    d = False
    for i in datafile.index:
        if i == date:
            d = True
            break

    if (d == False):
        return print('The date ' + date + ' is not in the file.')

    #remove the date from the datafiles
    datafile = datafile.drop(date, axis = 0)
    
    datafile = datafile.reset_index()

    np.save(LYADFILE / 'LyAdata24.npy', datafile)

    return print('The date ' + date + ' has been removed from the LyAdata24 file.')


def clusters_data(date):
    '''
    
    Create the clusters data file
    
    Parameters
    ------------
    date: date which we want to create the file in the format yy_mm_dd
    
    '''
    cdata = Lpicday(date).clusters_data()
    
    return
    

def LyA_addDate(date, corr_df = pd.read_csv('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/correction_arrays.csv').fillna(-100)):
    '''
    Add the data from Ldate.get_events_from_day and Ldaypic.clusters_day to a file (if the date is not already in the file) and create a new file if there is not one already

    Parameters
    ------------
    date: date which we want to add in the format yy_mm_dd

    '''
    if not (LYASAVE / date / 'LyA_data').exists():
        Path.mkdir(LYASAVE / date / 'LyA_data')
    
    Ld = Ldate(date, corr_df = corr_df)
    Lw = wf(date)
    
    #make sure that there are files to use for the current date
    if (len(Ld.get_filepaths()) == 0 and len(Lw.filelist()) == 0):
        return print('There are no LyA or waveform files for ' + date + '.')

    newfile = False #track we we created a new datafile24
    
    #if LyAdata24.txt does not exist, create a new one
    if (LYADFILE / 'LyAdata24.npy').is_file() == False:
        newfile = True
        datafile = pd.DataFrame([[0]*19], columns = cols)
    
    #if there already is a LyAdata24.txt file, load it
    else:
        datafile = read_df() #get the datafile

    #check that date2an is not already in the file
    for i in datafile['Date']:
        if i == date:
            print('The date ' + date +  ' has already been added to the LyAdata24 file.')
            return

    #the data that we want to add
    trc = Ld.peaks_data() #data from Ldate
    wav = Lw.wf_data() #data from wf
    summary = loadShortSummary(date) #short summary from the data loader
    
    #make sure that all the waveform files in wav are in the short summary
    keep = []
    summary_wf = list(summary['Waveform_12bit'])

    for i in wav['MCP5_Waveform']:
        if i in summary_wf:
            keep += [True]
        else:
            keep += [False]
    wav = wav.loc[keep]
    
    #make sure that all the LyA trc files in trc are in the short summary
    keep = []
    summary_ly = list(summary['LyA'])

    for i in trc['LyA']:
        if i in summary_ly:
            keep += [True]
        else:
            keep += [False]
    trc = trc.loc[keep]

    
    #we need to put the LyA files and the waveform files that belong to the same trigger in the same row
    #to do this we get the indices from the summary for the rows that match a LyA file and then set those indices in trc to be those
    i1 = summary.index[[str(val) != 'None' for val in summary['LyA']]] #indices in the summary that correspond to the rows in trc
    trc.set_index(i1, inplace = True) #set the indices of trc to i1
    
    #repeat the same for wav
    i2 = summary.index[[str(val) != 'None' for val in summary['Waveform_12bit']]] #indices in the summary that correspond to the rows in wav
    wav.set_index(i2, inplace = True) #set the indices of wav to i2
    
    #remove the run number columns from both dataframes, we get the run numbers from the summary
    trc = trc.drop('run', axis = 1)
    wav = wav.drop('run', axis = 1)
    
    #also need to drop NE50_I from wav and microwave from trc, so we can get those values for all events from the summary
    wav = wav.drop('NE50_I', axis = 1)
    trc = trc.drop('microwave', axis = 1)
    
    #concat trc and wav
    data = pd.concat([trc, wav], axis = 1)
    
    #we want to have a column with the date that the file of the row was created
    data.insert(0, 'Date', [date] * len(data))
    
    MCP5 = []
    time = []
    run = []
    ne50 = []
    mw = []
    for i in range(len(data)):
        curr = data.iloc[i]
        curr_sum = summary.iloc[data.iloc[i].name]
        time += [curr_sum['Time']]
        run += [curr_sum['run']]
        ne50 += [curr_sum['NE50_I']]
        if curr_sum['MW_power'] > 0.0001: mw += ['on']
        else: mw += ['off']
        if str(curr['MCP5_Waveform']) != 'nan':
            MCP5 += list(summary[summary['Waveform_12bit'] == curr['MCP5_Waveform']]['MCP5'])
        else:
            MCP5 += ['None']
    
    data.insert(0, 'NE50_I', ne50)
    data.insert(0, 'run', run)
    data.insert(0, 'MCP5_picture', MCP5)
    data.insert(0, 'microwave', mw)
    data['Time'] = time
    
    data = data[cols].sort_values(['Time'])
    
    data = data.fillna('None')
    data = data.replace('NaN', 'None')
    
    datafile = pd.concat([datafile, data]).sort_values(['Time']) #add the data to the datafile and sort by the date

    if (newfile == True):
        datafile = datafile.reset_index(drop = True)
        datafile = datafile.drop([0])
    
    datafile = datafile.reset_index(drop = True)

    np.save(LYADFILE / 'LyAdata24.npy', datafile)

    return


def read_df(trange = None, version = None):
    '''
        
    Read the npy file and get a pandas dataframe with the correct types for the columns
    
    Parameters
    ------------
    version: default None, if version is something else, the datafile with the name 'LyAdata24_v' + str(version) + '.npy' will be read
        
    '''
    if trange == None: trange = [0, 10001]
    
    if version == None:
        df = pd.DataFrame(np.load(LYADFILE / ('LyAdata24.npy'), allow_pickle = True), columns = cols)
        
    else:
        df = pd.DataFrame(np.load(LYADFILE / ('LyAdata24_v' + str(version) + '.npy'), allow_pickle = True), columns = cols)
    
    return df
