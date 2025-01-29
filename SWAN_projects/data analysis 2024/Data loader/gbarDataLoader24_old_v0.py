#!/usr/bin/env python
# coding: utf-8

# In[113]:


#use this for date from the 24_04_30 to 24_05_14

import sys
sys.path.append('/eos/home-i00/l/lkoller/data-analysis-software')

from pathlib import Path
import numpy as np
import pandas as pd
import fnmatch, re
import pytimber
import matplotlib.pyplot as plt
from datetime import datetime
from time import mktime
from tqdm import tqdm
import lmfit
from readTrc_4CH import Trc
import gbarDataLoader as gd
import csv

'''
Replace the following three paths

At DATAFILE the datafile24 is saved
At DATASUMMARY a folder for the date will be created and the summary, short_summary and log are saved there
At ELENADATA a folder for the date should exists with the file of the elena info. The elena text file is also saved there.
At INFOS the runData24 file is saved
'''
############################################################################################
DATAFILE = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24'
DATASUMMARY = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'datasummary24' 
ELENADATA = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'elenadata24'
INFOS = Path('/eos') / 'user' / 'l' / 'lkoller' / 'SWAN_projects' / 'data analysis 2024' / 'Data loader' / 'infos'
############################################################################################

PROTONGUNPC = Path('/eos') / 'experiment' / 'gbar' /'pgunpc' / 'data'
RCPC = Path('/eos') / 'experiment' / 'gbar' / 'rcpc' / 'data'

#log file column names for the ones after 24_05_14
sequence_logFile_new = ['Time', 'run number', 'event number', 'DAQ', 'pbar trap', 'positron', 'Valve in', 'MCP in', 'MW power', #8 general info
                    'H-sou pressure','cube pressure','RC pressure','MW pressure','SwY pressure','LyA pressure', #14 pressures
                   'Main delay','PCOs delay','LyA delay','Laser delay', #18 delays
                   'Faraday cup current', 'MCP4 current', 'MCP5 current', #21 currents
                   'St_dec_up','St_dec_dw','St_dec_ri','St_dec_le','St_Cub_up','St_Cub_dw','St_Cub_ri','St_Cub_le', #29 dec and cub steerers
                   'St_RC1_up','St_RC1_dw','St_RC1_ri','St_RC1_le','St_RC2_up','St_RC2_dw','St_RC2_ri','St_RC2_le','St_RC3_up','St_RC3_dw','St_RC3_ri','St_RC3_le', #41 RC1, RC2, RC3 steerers
                   'St_Al1_up','St_Al1_dw','St_Al1_ri','St_Al1_le','St_Al2_up','St_Al2_dw','St_Al2_ri','St_Al2_le', #49 Al1, Al2 steerers
                   'QT_RC1_+','QT_RC1_-','QT_RC2_+','QT_RC2_-','QT_RC3_+','QT_RC3_-','SwY_1_+','SwY_1_-','SwY_3_+','SwY_3_-', #59 qt and swy voltages
                   'EL_pT_in','EL_pT_1','EL_pT_2','EL_pT_3','EL_pT_4','EL_RC_in','EL_SwY_+','EL_Al1_+','EL_Al2_+','EL_Al3_+', #69 EL voltages
                   'Ly_MCP_1','Ly_MCP_2','Ly_MCP_3','Ly_MCP_4','Qnch_+','Qnch_-','TgDefl_+','H-Defl_+','H-Cor1_-','H-Cor1-', #79 LyA MCPs bias, Quenching, TgDefl, H-Defl, H-Cor voltages
                   '1_phos_+','1_mcp_+','2_phos_+','2_mcp_+','3_phos_+','3_mcp_+','3.5_grid_-','3.5_phos_+','3.5_mcp_+','4_phos_+','4_mcp_+','5_phos_+','5_mcp_+','6_phos_+','6_mcp_+','7_phos_+','7_mcp_+','Sci_1_-','Sci_2_-', #98  mcp + sci voltages
                   'H_offs', 'target position', 'front bias', #101 general info
                   'mw_amp_curr', 'hfs_temp', 'hfs_freq', 'hfs_pow', 'sc_temp', 'sc_freq', 'sc_pow', #108 microwave parameters
                   'empty 1', 'empty 2', 'empty 3', 'empty 4', 'empty 5', 'empty 6', 'empty 7', 'empty 8'] #116 empty columns

#list with the names of the columns for the files in the pgungc/data/date/date.txt (replace date)
sequence_logFile = ['Time','Valve in','MCP in','MW power','run number', #4
                   'DAQ','H-sou pressure','cube pressure','RC pressure','MW pressure','SwY pressure','LyA pressure', #11
                   'Main delay','PCOs delay','LyA delay','Laser delay','FC current','RC current','SwY current', #19
                   'St_dec_up','St_dec_dw','St_dec_ri','St_dec_le','St_Cub_up','St_Cub_dw','St_Cub_ri','St_Cub_le', #27
                   'St_RC1_up','St_RC1_dw','St_RC1_ri','St_RC1_le','St_RC2_up','St_RC2_dw','St_RC2_ri','St_RC2_le','St_RC3_up','St_RC3_dw','St_RC3_ri','St_RC3_le', #39
                   'St_Al1_up','St_Al1_dw','St_Al1_ri','St_Al1_le','St_Al2_up','St_Al2_dw','St_Al2_ri','St_Al2_le', #47
                   'QT_RC1_+','QT_RC1_-','QT_RC2_+','QT_RC2_-','QT_RC3_+','QT_RC3_-','SwY_1_+','SwY_1_-','SwY_3_+','SwY_3_-', #57
                   'EL_pT_in','EL_pT_1','EL_pT_2','EL_pT_3','EL_pT_4','EL_RC_in','EL_SwY_+','EL_Al1_+','EL_Al2_+','EL_Al3_+', #67
                   'Ly_MCP_1','Ly_MCP_2','Ly_MCP_3','Ly_MCP_4','Qnch_+','Qnch_-','TgDefl_+','H-Defl_+','H-Cor1_-','H-Cor1-', #77
                   '1_phos_+','1_mcp_+','2_phos_+','2_mcp_+','3_phos_+','3_mcp_+','3.5_grid_-','3.5_phos_+','3.5_mcp_+','4_phos_+','4_mcp_+','5_phos_+','5_mcp_+','6_phos_+','6_mcp_+','7_phos_+','7_mcp_+','Sci_1_-','Sci_2_-', #96
                   'H_offs', 'target position', 'front bias', #99 general info
                   'mw_amp_curr', 'hfs_temp', 'hfs_freq', 'hfs_pow', 'sc_temp', 'sc_freq', 'sc_pow', #106 microwave parameters
                   'empty 1', 'empty 2', 'empty 3', 'empty 4', 'empty 5', 'empty 6', 'empty 7', 'empty 8'] #114 empty columns

#list with the names of the columns for the rcpc and pgunpc
sequence = ['MCP1', 'MCP2', 'MCP3', 'MCP3.5', 'MCP4', 'MCP5', 'MCP7', 'Waveform_12bit', 'CMOS_Tracker', 'DRS4',
                'Positron_CH1','Positron_CH2','Positron_CH3','Positron_CH4','SD','LyA','SD_LyA']

#list with the names of the columns for the data from elena
elena_sequence = ['Cy_Des','NE00_I','NE50_I', 'NE00_Bm1', 'NE00_Bm2', 'NE00_Bm3', 'NE50_Bm1', 'NE50_Bm2', 'Int']

##list with the order of the columns for the summary files
sequence_summary = ['Time','Datetime','run number','event number', 'MCP1', 'MCP2', 'MCP3', 'MCP3.5', 'MCP4', 'MCP5', 'MCP7', 'Waveform_12bit', 'CMOS_Tracker', 'DRS4',
                   'Positron_CH1','Positron_CH2','Positron_CH3','Positron_CH4','SD','LyA','SD_LyA','Cy_Des','NE00_I','NE50_I', 'NE00_Bm1', 'NE00_Bm2', 'NE00_Bm3', 'NE50_Bm1', 'NE50_Bm2', 'Int',
                   'Valve in', 'MCP in', 'MW power', 'DAQ', 'pbar trap', 'positron', 'H-sou pressure', 'cube pressure', 'RC pressure', 'MW pressure','SwY pressure','LyA pressure', 
                   'Main delay','PCOs delay','LyA delay','Laser delay', 
                   'Faraday cup current', 'MCP4 current', 'MCP5 current', 
                   'St_dec_up','St_dec_dw','St_dec_ri','St_dec_le','St_Cub_up','St_Cub_dw','St_Cub_ri','St_Cub_le', 
                   'St_RC1_up','St_RC1_dw','St_RC1_ri','St_RC1_le','St_RC2_up','St_RC2_dw','St_RC2_ri','St_RC2_le','St_RC3_up','St_RC3_dw','St_RC3_ri','St_RC3_le', 
                   'St_Al1_up','St_Al1_dw','St_Al1_ri','St_Al1_le','St_Al2_up','St_Al2_dw','St_Al2_ri','St_Al2_le', 
                   'QT_RC1_+','QT_RC1_-','QT_RC2_+','QT_RC2_-','QT_RC3_+','QT_RC3_-','SwY_1_+','SwY_1_-','SwY_3_+','SwY_3_-', 
                   'EL_pT_in','EL_pT_1','EL_pT_2','EL_pT_3','EL_pT_4','EL_RC_in','EL_SwY_+','EL_Al1_+','EL_Al2_+','EL_Al3_+', 
                   'Ly_MCP_1','Ly_MCP_2','Ly_MCP_3','Ly_MCP_4','Qnch_+','Qnch_-','TgDefl_+','H-Defl_+','H-Cor1_-','H-Cor1-', 
                   '1_phos_+','1_mcp_+','2_phos_+','2_mcp_+','3_phos_+','3_mcp_+','3.5_grid_-','3.5_phos_+','3.5_mcp_+','4_phos_+','4_mcp_+','5_phos_+','5_mcp_+','6_phos_+','6_mcp_+','7_phos_+','7_mcp_+','Sci_1_-','Sci_2_-', 
                   'H_offs', 'target position', 'front bias', 
                   'mw_amp_curr', 'hfs_temp', 'hfs_freq', 'hfs_pow', 'sc_temp', 'sc_freq', 'sc_pow', 
                   'empty 1', 'empty 2', 'empty 3', 'empty 4', 'empty 5', 'empty 6', 'empty 7', 'empty 8']
    

def loadVIS():
    '''
    
    Set some general rules how dataframes and plots are displayed
    
    '''
    pd.set_option("display.max_columns",None)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.spines.left'] = True #False
    plt.rcParams['axes.spines.right'] = True #False
    plt.rcParams['axes.spines.top'] = True #False
    plt.rcParams['axes.spines.bottom'] = True #False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.labelcolor'] = '#555555'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['figure.figsize'] = 6,4
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['figure.titleweight'] = 'normal'
    plt.rcParams['font.family'] = 'sans-serif'
    

def loadRun():
    '''
    
    Put the run data starting from the 24_04_30 in a dataframe and save it at the path DATAFILE
    
    '''
    runc = list(csv.reader(list(open(PROTONGUNPC / 'Logs'/ 'DAQlog.txt', 'r', encoding='windows-1252' )), delimiter = '\t'))
    runc = [val for val in runc if int(val[0][:10]) >= 1714463780]
    runc = [val + bal for val,bal in zip(runc[::2],runc[1::2])]
    
    for i in ['Start_run: ', 'Set_name: ', 'Comment:  ', 'Stop-Evts: ', 'Set_name: ']:
        [val.remove(i) for val in runc]
        
    runf = pd.DataFrame([val[:-1] for val in runc], columns = ['start run timestamp', 'start run datetime', 'run number', 'setup', 'comment', 'stop run timestamp', 'stop run datetime', 'events'])
    runf.to_csv(str(INFOS / ('runData24.txt')), sep='\t', index=False)
    
    return pd.read_csv(str(INFOS) + '/runData24.txt', delimiter = '\t')
    
    

def loadLog(date2an):
    '''

    Load the log file from the protongun PC and label the columns
    Does not work for date2an = 24_04_25 and before
    
    Parameters
    ------------
    date2an = the date for which we want to create a log file
    
    '''
    if ((DATASUMMARY / date2an / ('log_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('log_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('LOG file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('log_'+date2an+'.txt')), sep='\t')
    else:
        if not (PROTONGUNPC / date2an).exists() or not (PROTONGUNPC / date2an / (date2an+'.txt')).is_file():
            return print('Wrong date formate or date is not uploaded yet.')     
        else:
            if not (DATASUMMARY/ date2an).exists():
                Path.mkdir(DATASUMMARY / date2an) 
            
            logl = list(open(PROTONGUNPC / date2an / (date2an + '.txt'), 'r', ))
            logc = list(csv.reader(logl, delimiter = '\t'))
            
            if set([len(val) for val in logc]) == {114, 115}:
                [val if len(val) == 114 else val.pop(16) for val in logc]
                
            runf = loadRun()
                        
            log = pd.DataFrame(logc, dtype = float, columns = sequence_logFile)
            
            crunset = sorted(set(log['run number']))
            runset = set(runf['run number'])
            
            for i in crunset:
                rnum = float(i)
                
                if not (rnum in runset):
                    log.loc[list(log.loc[:, 'run number'] == i), 'run number'] = 0
                
                else:
                    if not (rnum + 1 in runset):
                        log.loc[list([val > list(runf['stop run timestamp'][runf['run number'] == rnum])[0] for val in log['Time']]), 'run number'] = 0
                    
                    else:
                        log.loc[list([list(runf['start run timestamp'][runf['run number'] == rnum + 1])[0] > val > list(runf['stop run timestamp'][runf['run number'] == rnum])[0] for val in log['Time']]), 'run number'] = 0
            '''
            for i in crunset:
                rnum = float(i)
                
                if not (rnum in runset):
                    log['run number'][log['run number'] == i] = 0
                
                else:
                    if not (rnum + 1 in runset):
                        log['run number'][[val > list(runf['stop run timestamp'][runf['run number'] == rnum])[0] for val in log['Time']]] = 0
                    
                    else:
                        log['run number'][[list(runf['start run timestamp'][runf['run number'] == rnum + 1])[0] > val > list(runf['stop run timestamp'][runf['run number'] == rnum])[0] for val in log['Time']]] = 0'''
            
            #put all columns that are in sequence_logFile_new that are not in sequence_logFile into log with values 0
            for i in range(len(sequence_logFile_new)):
                if not sequence_logFile_new[i] in list(log.columns):
                    log.insert(0, sequence_logFile_new[i], [0] * len(log))
            
            log = log[sequence_logFile_new]

            log.to_csv(str(DATASUMMARY / date2an / ('log_'+date2an+'.txt')), sep='\t', index=False)
            return log
        


def loadRCPC(date2an):
    '''
    
    Get the times and locations for the files from the different MCPs which are in the RCPC folder
    
    Parameters
    ------------
    date2an = the date for which we want to create a log file
    
    Returns
    ------------
    dataframe with the times and filenames for the MCPSs at date2an sorted by time
    
    '''
    mcp1_file = []
    mcp1_time = []
    mcp2_file = []
    mcp2_time = []
    mcp3_file = []
    mcp3_time = []
    mcp35_file = []
    mcp35_time = []
    mcp4_file = []
    mcp4_time = []
    mcp7_file = []
    mcp7_time = []
    
    if (RCPC / date2an).exists() == False:
        print('RCPC files might not have been uploaded yet.')
    else:
        p = RCPC / date2an
        for file in p.iterdir():
            if (fnmatch.fnmatch(file, '*MCP1*') or fnmatch.fnmatch(file, '*MCP-p1*')) and fnmatch.fnmatch(file, '*.tif') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp1_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp1_file.append(file)
            if (fnmatch.fnmatch(file, '*MCP2*') or fnmatch.fnmatch(file, '*MCP-p2*')) and fnmatch.fnmatch(file, '*.tif') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp2_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp2_file.append(file)
            if (fnmatch.fnmatch(file, '*MCP3*') or fnmatch.fnmatch(file, '*MCP-p3*')) and fnmatch.fnmatch(file, '*.tif') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp3_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp3_file.append(file)
            if (fnmatch.fnmatch(file, '*MCP3.5*') or fnmatch.fnmatch(file, '*MCP-p3.5*')) and fnmatch.fnmatch(file, '*.tif') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp35_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp35_file.append(file)
            if (fnmatch.fnmatch(file, '*PCO-ReC*') or fnmatch.fnmatch(file, '*MCP5*')) and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp4_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp4_file.append(file)
            if (fnmatch.fnmatch(file, '*VCXG-51M-7*') and fnmatch.fnmatch(file, '*.tif')) and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp7_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp7_file.append(file)
        
    df1 = pd.DataFrame({'Time' : mcp1_time, 'MCP1' : mcp1_file})
    df2 = pd.DataFrame({'Time' : mcp2_time, 'MCP2' : mcp2_file})
    df3 = pd.DataFrame({'Time' : mcp3_time, 'MCP3' : mcp3_file})
    df35 = pd.DataFrame({'Time' : mcp35_time, 'MCP3.5' : mcp35_file})
    df4 = pd.DataFrame({'Time' : mcp4_time, 'MCP4' : mcp4_file})
    df7 = pd.DataFrame({'Time' : mcp7_time, 'MCP7' : mcp7_file})  
    return pd.concat([df1,df2,df3,df35,df4,df7], ignore_index=True).sort_values('Time', ascending=True, ignore_index=True)
        

def loadPGUNPC(date2an):
    '''
    
    files from the PGUNPC from date2an into a dataframe
    
    df5 are the locations of the pictures taken by MCP5
    dfwf are the locations of the waveform files
    dfcmos are the locations cmos detector
    dfdrs is deinstalled now
    dfposi1 are the locations positron waveform 1
    dfposi2 are the locations positron waveform 2
    dfposi3 are the locations positron waveform 3
    dfposi4 are the locations positron waveform 4
    dfsdposi are the locations png of the oscilloscope for the positrons
    dflya are the locations of the trc files with the data of the 4 channels from the lyman alpha setup
    dfsdlya are the locations png of the oscilloscope for the lya
    
    Returns
    ------------
    dataframe with all the data from above sorted by time. The columns are:
        df5 are the locations of the pictures taken by MCP5
        dfwf are the locations of the waveform files
        dfcmos are the locations cmos detector
        dfdrs is deinstalled now
        dfposi1 are the locations positron waveform 1
        dfposi2 are the locations positron waveform 2
        dfposi3 are the locations positron waveform 3
        dfposi4 are the locations positron waveform 4
        dfsdposi are the locations png of the oscilloscope for the positrons
        dflya are the locations of the trc files with the data of the 4 channels from the lyman alpha setup
        dfsdlya are the locations png of the oscilloscope for the lya    
    
    '''
    mcp5_time, mcp5_file = [],[]
    waveform_time, waveform_file = [],[]
    cmos_time, cmos_file = [],[]
    drs_time, drs_file = [],[]
    posi_c1_time, posi_c1_file = [],[]
    posi_c2_time, posi_c2_file = [],[]
    posi_c3_time, posi_c3_file = [],[]
    posi_c4_time, posi_c4_file = [],[]
    sd_posi_time, sd_posi_file = [],[]
    lya_time, lya_file = [],[]
    sd_lya_time, sd_lya_file = [],[]
    
    if (PROTONGUNPC / date2an).exists() == False:
        print('PGunPC files might not have been uploaded yet.')
    else:
        p = PROTONGUNPC / date2an
        posi = p / (date2an+'posi')
        lya = p / (date2an+'lya')
        for file in p.iterdir():
            if (fnmatch.fnmatch(file, '*PCO-SwY*') or fnmatch.fnmatch(file, '*MCP6*')) and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                mcp5_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp5_file.append(file)
            if fnmatch.fnmatch(file, '*WF1234*') and fnmatch.fnmatch(file,'*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                waveform_time.append(int(re.search('WF1234\.(.*?)\.', str(file)).group(1)))
                waveform_file.append(file)
            if fnmatch.fnmatch(file, '*VCXG-51M-5*') and fnmatch.fnmatch(file, '*.tif') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                cmos_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                cmos_file.append(file)
            if fnmatch.fnmatch(file, '*drs4*') and fnmatch.fnmatch(file, '*.txt') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                filetime = re.search('drs4\_(.*?)\.', str(file)).group(1)
                drs_time.append(int(mktime(datetime(int('20'+filetime[0:2]),int(filetime[3:5]),int(filetime[6:8]),int(filetime[9:11]), int(filetime[12:14]), int(filetime[15:17])).timetuple())))
                drs_file.append(file)
        if (posi.exists()):
            for file in posi.iterdir():
                if fnmatch.fnmatch(file, '*C1*') and fnmatch.fnmatch(file, '*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    posi_c1_time.append(int(re.search('C1\.(.*?)\.', str(file)).group(1)))
                    posi_c1_file.append(file)
                if fnmatch.fnmatch(file, '*C2*') and fnmatch.fnmatch(file, '*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    posi_c2_time.append(int(re.search('C2\.(.*?)\.', str(file)).group(1)))
                    posi_c2_file.append(file) 
                if fnmatch.fnmatch(file, '*C3*') and fnmatch.fnmatch(file, '*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    posi_c3_time.append(int(re.search('C3\.(.*?)\.', str(file)).group(1)))
                    posi_c3_file.append(file)
                if fnmatch.fnmatch(file, '*C4*') and fnmatch.fnmatch(file, '*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    posi_c4_time.append(int(re.search('C4\.(.*?)\.', str(file)).group(1)))
                    posi_c4_file.append(file)
                if fnmatch.fnmatch(file, '*SD*') and fnmatch.fnmatch(file, '*.png') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    sd_posi_time.append(int(re.search('SD\.(.*?)\.', str(file)).group(1)))
                    sd_posi_file.append(file)
        if (lya.exists()):
            for file in lya.iterdir():
                if fnmatch.fnmatch(file, '*LY*') and fnmatch.fnmatch(file, '*.trc') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    lya_time.append(int(re.search('LY1234\.(.*?)\.', str(file)).group(1)))
                    lya_file.append(file)
                """
                if fnmatch.fnmatch(file, '*C2*') and fnmatch.fnmatch(file, '*.trc'):
                    lya_c2_time.append(int(re.search('C2\.(.*?)\.', str(file)).group(1)))
                    lya_c2_file.append(file) 
                if fnmatch.fnmatch(file, '*C3*') and fnmatch.fnmatch(file, '*.trc'):
                    lya_c3_time.append(int(re.search('C3\.(.*?)\.', str(file)).group(1)))
                    lya_c3_file.append(file)
                if fnmatch.fnmatch(file, '*C4*') and fnmatch.fnmatch(file, '*.trc'):
                    lya_c4_time.append(int(re.search('C4\.(.*?)\.', str(file)).group(1)))
                    lya_c4_file.append(file)
                """
                if fnmatch.fnmatch(file, '*SD*') and fnmatch.fnmatch(file, '*.png') and fnmatch.fnmatch(file, '*.sys.v#.*') == False:
                    sd_lya_time.append(int(re.search('SD\.(.*?)\.', str(file)).group(1)))
                    sd_lya_file.append(file)
    df5 = pd.DataFrame({'Time' : mcp5_time, 'MCP5' : mcp5_file})
    dfwf = pd.DataFrame({'Time' : waveform_time, 'Waveform_12bit' : waveform_file})
    dfcmos = pd.DataFrame({'Time' : cmos_time, 'CMOS_Tracker' : cmos_file})
    dfdrs = pd.DataFrame({'Time' : drs_time, 'DRS4'  : drs_file})
    dfposi1 = pd.DataFrame({'Time' : posi_c1_time, 'Positron_CH1' : posi_c1_file})
    dfposi2 = pd.DataFrame({'Time' : posi_c2_time, 'Positron_CH2' : posi_c2_file})
    dfposi3 = pd.DataFrame({'Time' : posi_c3_time, 'Positron_CH3' : posi_c3_file})
    dfposi4 = pd.DataFrame({'Time' : posi_c4_time, 'Positron_CH4' : posi_c4_file})
    dfsdposi = pd.DataFrame({'Time' : sd_posi_time, 'SD':sd_posi_file})
    dflya = pd.DataFrame({'Time' : lya_time, 'LyA' : lya_file})
    #dflya2 = pd.DataFrame({'Time' : lya_c2_time, 'LyA_CH2' : lya_c2_file})
    #dflya3 = pd.DataFrame({'Time' : lya_c3_time, 'LyA_CH3' : lya_c3_file})
    #dflya4 = pd.DataFrame({'Time' : lya_c4_time, 'LyA_CH4' : lya_c4_file})
    dfsdlya = pd.DataFrame({'Time' : sd_lya_time, 'SD_LyA':sd_lya_file})
    return pd.concat([df5, dfwf, dfcmos, dfdrs, dfposi1, dfposi2, dfposi3, dfposi4, dfsdposi, dflya, dfsdlya], ignore_index=True).sort_values('Time', ascending=True, ignore_index=True)

def loadElena(date2an):
    '''
    
    Get the parameters for the Elena/AD runs on the day date2an and return them in a dataframe
    
    Parameters
    ------------
    date2an = the date from which we want the data
    
    Returns
    ------------
    elena = dataframe with the data for the elena runs on date2an, the colums are:
        Time = unix timestamp of the cycle
        Cy_Des = description of the cycle parameters
        NE00_I = NE00 Intensity
        NE50_I = NE50 Intensity
        NE00_Bm1 = Whether or not the first beam monitor for the NE00 line is in
        NE00_Bm2 = Whether or not the second beam monitor for the NE00 line is in
        NE00_Bm3 = Whether or not the third beam monitor for the NE00 line is in
        NE50_Bm1 = Whether or not the first beam monitor for the NE50 line is in
        NE50_Bm2 = Whether or not the second beam monitor for the NE50 line is in
        Int = Total intensity
        Linac_Rad = Radiation monitor from the linac
        
    '''
    if ((ELENADATA / date2an / ('elena_' + date2an + '.txt')).is_file() == True):
        print('Elena file for ' + date2an + ' already exists.')
        return pd.read_csv(str(ELENADATA / date2an / ('elena_'+date2an+'.txt')), sep='\t')
    
    if not (ELENADATA / date2an / ('ELENA_BM_' + date2an + '.csv')).is_file():
        return print('The elena data for ' + date2an + ' has not been downloaded yet. Please run the ELENA_daily_data2test_file notebook.')
    
    bm = pd.read_csv(ELENADATA / date2an / ('ELENA_BM_' + date2an + '.csv'), header = 0, 
                     names = ['Time','NE00_I','NE50_I', 'NE00_Bm1', 'NE00_Bm2', 'NE00_Bm3', 'NE50_Bm1', 'NE50_Bm2', 'Int'])
    cy = pd.read_csv(ELENADATA / date2an / ('ELENA_Cycles_' + date2an + '.csv'), header = 0, names = ['Time', 'Cy_Des'])
    #lac = pd.read_csv(ELENADATA / date2an / ('Linac_RadMon_' + date2an + '.csv'), header = 0, names = ['Time', 'Linac_Rad'])
    
    if (bm.shape[0] != cy.shape[0]):
        return print('cycle and beam data does not match')
    
    bm = bm.drop(['Time'], axis = 1)
    
    elena = pd.concat([cy,bm], axis = 1).round({'Time':0}).sort_values(by = ['Time'])
    
    elena['Time'] = elena['Time'] + 12
    
    elena.to_csv(str(ELENADATA / date2an / ('elena_'+date2an+'.txt')), sep='\t', index=False)
    
    return elena


def loadDatafile(date2an):
    '''
    
    match all the data from elena, pgunpc and rcpc to the correct event in the log file
    
    '''
    if ((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d') ):
        print('SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('summary_'+date2an+'.txt')), sep='\t')
    else:
        if not (DATASUMMARY/ date2an).exists():
            Path.mkdir(DATASUMMARY / date2an) 
        dfpgun = loadPGUNPC(date2an)
        dfrcpc = loadRCPC(date2an)
        dflog = loadLog(date2an)
        
        strucdf = pd.DataFrame([[0]*144], columns = sequence_summary)

        datafiles= pd.concat([dfpgun, dfrcpc], ignore_index=True).round({'Time':0}).sort_values(['Time'])
        
        elena = loadElena(date2an)
        
        cont = True #track for datafiles if we can continue (for some reason break does not work?)
        pos = 0
        epos = 0
        for idx, timestamp in tqdm(enumerate(dflog['Time']), total = len(dflog)):
            
            for eidy, ets in enumerate(elena['Time'][epos:]):
                if (abs(timestamp - ets) < 7):
                    epos = eidy + epos
                    for elabel in elena_sequence:
                        dflog.loc[idx, elabel] = elena[elabel][epos]
                    break
                    
            if not (timestamp - datafiles['Time'][0]) < -7:
                for idy, ts in enumerate(datafiles['Time'][pos:]):
                    if abs(timestamp - ts) < 7:
                        pos = idy + pos
                        break
                while(cont == True and abs(datafiles['Time'][pos] - timestamp) < 7):
                    for label in sequence:
                        if (pd.isna(datafiles[label][pos]) == False): 
                            dflog.loc[idx, label] = datafiles[label][pos]
                    pos += 1
                    if (pos == len(datafiles)):
                        cont = False

        dflog = pd.concat([dflog, pd.DataFrame({'Datetime' : [datetime.fromtimestamp(entry) for entry in dflog['Time']]})], axis=1)
        df_final = pd.DataFrame()
        df_final = pd.concat([strucdf,dflog]).drop([0]).reset_index(drop = True)
        df_final.to_csv(str(DATASUMMARY / date2an / ('summary_'+date2an+'.txt')), sep='\t', index=False)
        
    return df_final


def loadShortSummary(date2an):  
    '''
    
    make a document 'shortSum_date2an.txt' which only contains the rows of summary_date2an.txt which have a non NaN value somewhere in rcpc or pgunpc
    
    '''
    if ((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).is_file() == True):# and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('SHORT SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')), sep='\t')
    else: 
        longData = loadDatafile(date2an)
        elena = loadElena(date2an)
        log = loadLog(date2an)

        #drop all rows that have only nan for the pgunpc and rcpc (the threshold len(sequence_logFile) + len(elena_sequence) + 3 should be correct, not 100% sure)
        shortData = longData.dropna(thresh=len(sequence_logFile_new) + len(elena_sequence) + 2).reset_index(drop = True)
        shortData.to_csv(str(DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')), sep='\t', index = False)

    return shortData


def addDate(date2an):
    '''
    
    add the short summary from date2an to datafile24.txt
    if no datafile24.txt exists, we create a new one
    
    '''
    newfile = False #track we we created a new datafile24
    
    #if datafile24.txt does not exist, create a new one
    if (DATAFILE / 'datafile24.txt').is_file() == False:
        newfile = True
        datafile = pd.DataFrame([['00_00_00'] + [1]*144], columns = ['Date'] + sequence_summary)

    else:
        datafile = pd.read_csv(str(DATAFILE) + '/datafile24.txt', delimiter = '\t') #get the datafile

    #check that date2an isnt already in the file
    for i in datafile['Date']:
        if i == date2an:
            return print('The date ' + date2an + ' has already been added to the datafile.')
        
    summary = loadShortSummary(date2an) #load the summary for date2an, which we want to add to the file 
    summary.insert(0, 'Date', [date2an] * len(summary))
    
    datafile = pd.concat([datafile, summary]).sort_values(by = ['Time'], axis = 0).reset_index(drop = True) #add the summary to the datafile and sort by the date
    
    if (newfile == True):
        datafile = datafile.drop([0])
        
    datafile.to_csv(str(DATAFILE) + '/datafile24.txt', sep='\t', index = False)
    
    return


def removeDate(date2an):
    '''
    
    remove the data from date2an from the datafile24.txt
    
    '''
    #if datafile24.txt does not exist, we just stop
    if (DATAFILE / 'datafile24.txt').is_file() == False:
        return print('The datafile does not exist yet.')
    
    #get the datafile
    datafile = pd.read_csv(str(DATAFILE) + '/datafile24.txt', delimiter = '\t', index_col = [0])
    
    #check if the date is in the datafile
    date_exists = False 
    for i in datafile.index:
        if i == date2an:
            date_exists = True
            break
    
    #if the date is not in the datafile, we can stop
    if (date_exists == False):
        return print('The date ' + date2an + ' is not in the datafile.')
    
    #remove the date from the datafiles
    datafile = datafile.drop(date2an, axis = 0)
    
    datafile.to_csv(str(DATAFILE) + '/datafile24.txt', sep='\t') #save the datafile with the removed date
    
    return print('The date ' + date2an + ' has been removed from the datafile.')

