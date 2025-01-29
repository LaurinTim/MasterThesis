from pathlib import Path
import pandas as pd
import fnmatch, re, math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, find_peaks_cwt
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import sparse
from scipy.sparse.linalg import spsolve

PROTONGUNPC = Path('/eos') / 'experiment' / 'gbar' /'pgunpc' / 'data'
RCPC = Path('/eos') / 'experiment' / 'gbar' / 'rcpc' / 'data'
ELENADATA = Path('/eos') / 'experiment' / 'gbar' / 'elena' / 'data'
DATASUMMARY = Path('/eos') / 'experiment' / 'gbar' / 'datasummary' / 'data'

def loadVIS():
    pd.set_option("display.max_columns",None)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.labelcolor'] = '#555555'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['figure.figsize'] = 6,4
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.titleweight'] = 'normal'
    plt.rcParams['font.family'] = 'sans-serif'


def getImg(file):
    if pd.isna(file)==True:
        return np.array([np.zeros(900) for i in range(900)])
    elif fnmatch.fnmatch(file, '*SwY*'):
        ped_img = np.array(plt.imread("/eos/experiment/gbar/pgunpc/data/22_10_17/PCO-SwY_exp_1_us_1666003522.001.ped.tif")).astype(float)
        img = np.array(plt.imread(file)).astype(float)
        img -= ped_img

        return img
    else:
        img = np.array(plt.imread(file)).astype(float)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] < 850:
                    img[x][y] = 0
        return img

def clusterFinder(img):
    thresh = abs(img.mean())+3*img.std()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (y-740)**2/525**2+(x-525)**2/510**2 > 1:
                img[x][y] = 0
    blobs = blob_log(img, min_sigma=1, threshold=thresh)
    blobs[:,2] = blobs[:,2]*np.sqrt(2)
    return blobs

def getWave(file):
    if pd.isna(file)==True:
        wave_time = np.linspace(0,10,5002)
        df = pd.DataFrame({'Scint_RC':np.zeros(5002), 'MCP_RC':np.zeros(5002), 'MCP_Swy':np.zeros(5002), 'Scint_Swy':np.zeros(5002)})
        return wave_time, df
    else:
        wave_time = np.linspace(0,10,5002)
        wave = pd.read_csv(file, sep='\t', header=None, skiprows=5)
        wave = wave.rename(columns={0:'time', 1:'Scint_RC', 2:'MCP_RC', 3:'MCP_Swy', 4:'Scint_Swy'})

        return wave_time, wave

def getSwyWave(file):
    if pd.isna(file) == True:
        steps = 10/5002
        wave_time = np.linspace(0,10,5002) # 10us
        ped_time = math.ceil(1/steps)
        
        swy1 = math.floor(2.8/steps)
        swy2 = math.ceil(4.2/steps)
        wave_time_swy = np.linspace(swy1*steps, swy2*steps, swy2-swy1)
        df = pd.DataFrame({'MCP_Swy': np.zeros(len(wave_time_swy))})
        peaks = []
        return wave_time_swy, df, peaks
    else:
        wave = pd.read_csv(file, sep='\t', header=None, skiprows=5)
        wave = wave.rename(columns={0:'time', 1:'Scint_RC', 2:'MCP_RC', 3:'MCP_Swy', 4:'Scint_Swy'})

        steps = 10/5002
        wave_time = np.linspace(0,10,5002) # 10us
        ped_time = math.ceil(1/steps)
        
        mcp_swy_ped_std = wave.MCP_Swy[0:ped_time].std()
        mcp_swy_ped_max = wave.MCP_Swy[0:ped_time].max()
        mcp_swy_ped_min = wave.MCP_Swy[0:ped_time].min()
        swy1 = math.floor(4.8/steps)
        swy2 = math.ceil(7/steps)
        wave_time_swy = np.linspace(swy1*steps, swy2*steps, swy2-swy1)
        peaks, properties = find_peaks(wave.MCP_Swy[swy1:swy2], height=(mcp_swy_ped_max+3*mcp_swy_ped_std))

        return wave_time_swy, wave['MCP_Swy'][swy1:swy2].reset_index(drop=True), peaks

def getRCWave(file):
    if pd.isna(file) == True:
        steps = 10/5002
        wave_time = np.linspace(0,10,5002) # 10us
        ped_time = math.ceil(1/steps)
        
        rc1 = math.floor(3/steps)
        rc2 = math.ceil(4.5/steps)
        wave_time_rc = np.linspace(rc1*steps, rc2*steps, rc2-rc1)
        df = pd.DataFrame({'Scint_RC': np.zeros(len(wave_time_rc)), 'MCP_RC' : np.zeros(len(wave_time_rc)), 'Scint_Swy':np.zeros(len(wave_time_rc))})
        return wave_time_rc, df
    else:
        wave = pd.read_csv(file, sep='\t', header=None, skiprows=5)
        wave = wave.rename(columns={0:'time', 1:'Scint_RC', 2:'MCP_RC', 3:'MCP_Swy', 4:'Scint_Swy'})

        steps = 10/5002
        wave_time = np.linspace(0,10,5002) # 10us
        ped_time = math.ceil(1/steps)
        
        rc1 = math.floor(1/steps)
        rc2 = math.ceil(2.5/steps)
        wave_time_rc = np.linspace(rc1*steps, rc2*steps, rc2-rc1)
        return wave_time_rc, wave[{'Scint_RC','MCP_RC','Scint_Swy'}][rc1:rc2]

# asymmetrxi least squares smoothing
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def getDRS(timestamp, file):
    if pd.isna(file) == True:
        drs_time = np.linspace(0,1,1024)
        df = pd.DataFrame({'Channel1' : np.zeros(1024), 'Channel2' : np.zeros(1024), 'Channel3' : np.zeros(1024), 'Channel4':np.zeros(1024), 'Channel5' : np.zeros(1024),
                           'ch1_smooth' : np.zeros(1024), 'ch2_smooth' : np.zeros(1024), 'ch3_smooth' : np.zeros(1024), 'ch4_smooth':np.zeros(1024), 'ch5_smooth' : np.zeros(1024),
                           'ch1_corr' : np.zeros(1024), 'ch2_corr' : np.zeros(1024), 'ch3_corr' : np.zeros(1024), 'ch4_corr':np.zeros(1024), 'ch5_corr' : np.zeros(1024)})
        df_peaks = pd.DataFrame({'Time':[0],'Peak1':[0], 'Height1':[0], 'Peak2':[0], 'Height2':[0], 'Peak3':[0], 'Height3':[0], 'Peak4':[0], 'Height4':[0], 'Peak5':[0], 'Height5':[0]})

        return drs_time, df, df_peaks
    else:
        drs = pd.read_csv(file, sep='\t')
        drs_time = np.linspace(0,1024,1024) # 1us
        ch1_smooth = pd.DataFrame({'ch1_smooth':baseline_als(-drs.Channel1, 100000,0.01)}) #changed from 1000, 0.1 for line fit
        ch2_smooth = pd.DataFrame({'ch2_smooth':baseline_als(-drs.Channel2, 100000,0.01)})
        ch3_smooth = pd.DataFrame({'ch3_smooth':baseline_als(-drs.Channel3, 100000,0.01)})
        ch4_smooth = pd.DataFrame({'ch4_smooth':baseline_als(-drs.Channel4, 100000,0.01)})
        ch5_smooth = pd.DataFrame({'ch5_smooth':baseline_als(-drs.Channel5, 100000,0.01)})
        drs = pd.concat([drs, ch1_smooth,ch2_smooth,ch3_smooth,ch4_smooth,ch5_smooth], axis=1)

        ch1_corr = pd.DataFrame({'ch1_corr': drs.Channel1+drs.ch1_smooth})
        ch2_corr = pd.DataFrame({'ch2_corr': drs.Channel2+drs.ch2_smooth})
        ch3_corr = pd.DataFrame({'ch3_corr': drs.Channel3+drs.ch3_smooth})
        ch4_corr = pd.DataFrame({'ch4_corr': drs.Channel4+drs.ch4_smooth})
        ch5_corr = pd.DataFrame({'ch5_corr': drs.Channel5+drs.ch5_smooth})
        drs = pd.concat([drs, ch1_corr,ch2_corr,ch3_corr,ch4_corr,ch5_corr], axis=1)

        peaks1= find_peaks_cwt(-drs.ch1_corr, widths=np.arange(10,150))
        peaks2= find_peaks_cwt(-drs.ch2_corr, widths=np.arange(10,150))
        peaks3= find_peaks_cwt(-drs.ch3_corr, widths=np.arange(10,150))
        peaks4= find_peaks_cwt(-drs.ch4_corr, widths=np.arange(10,150))
        peaks5= find_peaks_cwt(-drs.ch5_corr, widths=np.arange(10,150))

        df1 = pd.DataFrame({'Peak1':peaks1, 'Height1':drs.ch1_corr[peaks1]}).reset_index(drop=True)
        df2 = pd.DataFrame({'Peak2':peaks2, 'Height2':drs.ch2_corr[peaks2]}).reset_index(drop=True)
        df3 = pd.DataFrame({'Peak3':peaks3, 'Height3':drs.ch3_corr[peaks3]}).reset_index(drop=True)
        df4 = pd.DataFrame({'Peak4':peaks4, 'Height4':drs.ch4_corr[peaks4]}).reset_index(drop=True)
        df5 = pd.DataFrame({'Peak5':peaks5, 'Height5':drs.ch5_corr[peaks5]}).reset_index(drop=True)
        
        size = max(df1.shape[0],df2.shape[0],df3.shape[0],df4.shape[0],df5.shape[0])
        time = [timestamp for i in range(size)]
        dft = pd.DataFrame({'Time': time})
        df_peaks = pd.concat([dft,df1,df2,df3,df4,df5], axis=1)
        return drs_time, drs, df_peaks

def compareSwy(timestamp, img_file, wf_file):
    timestamp = datetime.fromtimestamp(timestamp)
    img = getImg(img_file)
    wave_time, wave, peaks = getSwyWave(wf_file)

    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,10))
    fig.suptitle(timestamp)
    
    ax1.set_title('Electric signal')
    ax1.plot(wave_time, wave, color='red', label='MCP Swy')
    ax1.scatter(wave_time[peaks], wave.MCP_Swy[peaks], color='dodgerblue', marker='x')
    ax1.set_xlabel('time [$\mu s$]')
    ax1.set_ylabel('electric signal [V]')
    ax1.set_ylim(bottom=-0.1, top=0.35)
    ax1.legend(loc='upper left')
    #[ax1.axvline(p, linestyle='-.', color='gray') for p in [3, 4]]

    ax2.imshow(img, cmap=plt.cm.viridis)
    if pd.isna(img_file) == True:
        title = 'No picture taken.'
    elif fnmatch.fnmatch(img_file, '*SwY*'):
        title = 'Switchyard: '
        patch = patches.Ellipse((740,525),width=1050, height=1020, transform=ax2.transData, color='tab:red', linewidth=3, alpha=1, fill=False)
        ax2.add_patch(patch)
    ax2.set_title(title)
    ax2.set_xlabel('y [pixels]')
    ax2.set_ylabel('z [pixels]')
    plt.tight_layout()
    plt.show()

    return 1

def compareRC(timestamp, img_file, wf_file):
    timestamp = datetime.fromtimestamp(timestamp)
    img = getImg(img_file)
    wave_time, wave = getRCWave(wf_file)
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,10))
    fig.suptitle(timestamp)
    
    ax1.set_title('Electric signal')
    ax1.plot(wave_time, wave['MCP_RC'], color='red', label='MCP RC')
    ax1.plot(wave_time, wave['Scint_RC'], color='orange', label='Scint RC')
    ax1.set_xlabel('time [$\mu s$]')
    ax1.set_ylabel('electric signal [V]')
    #ax1.set_ylim(bottom=-0.1, top=1.2)
    ax1.legend(loc='b')
    [ax1.axvline(p, linestyle='-.', color='gray') for p in [1.7, 2.1]]

    ax2.imshow(img, cmap=plt.cm.viridis)
    if pd.isna(img_file) == True:
        title = 'No picture taken.'
    elif fnmatch.fnmatch(img_file, '*ReC*'):
        title = 'Reaction chamber: '
        patch = patches.Ellipse((720,460),width=700, height=600, transform=ax2.transData, color='tab:red', linewidth=3, alpha=1, fill=False)
        ax2.add_patch(patch)
    ax2.set_title(title)
    ax2.set_xlabel('y [pixels]')
    ax2.set_ylabel('z [pixels]')
    plt.tight_layout()
    plt.show()

    return 1

def compareSwyDRS(timestamp, img_file, wf_file, drs_file, delay_PCO):
    timestamp = datetime.fromtimestamp(timestamp)
    img = getImg(img_file)
    wave_time, wave = getWave(wf_file)
    wave_time_swy, wave_swy, peaks_swy = getSwyWave(wf_file)
    drs_time, drs, peaks = getDRS(timestamp, drs_file)
    peaks1 = [int(item) for item in peaks.groupby('Time')['Peak1'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks2 = [int(item) for item in peaks.groupby('Time')['Peak2'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks3 = [int(item) for item in peaks.groupby('Time')['Peak3'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True] 
    peaks4 = [int(item) for item in peaks.groupby('Time')['Peak4'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks5 = [int(item) for item in peaks.groupby('Time')['Peak5'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    if pd.isna(delay_PCO) == True:
        delay_PCO =5
    elif delay_PCO == 0.001:
        delay_PCO = 5
    else:
        delay_PCO = ((delay_PCO*1e9-999430)/1000).round(4)

    fig, ([ax1,ax2],[ax3, ax4]) = plt.subplots(2,2, figsize=(10,10))
    fig.suptitle(timestamp)
    
    ax1.set_title('Waveform ')
    ax1.plot(wave_time, wave['MCP_RC'], color='indianred', label='MCP RC')
    ax1.plot(wave_time, wave['MCP_Swy'], color='dodgerblue', label='MCP Swy')
    ax1.plot(wave_time, wave['Scint_RC'], color='darkorange', label='Scint RC')
    ax1.plot(wave_time, wave['Scint_Swy'], color='limegreen', label='Scint Swy')
    ax1.set_xlabel('time [$\mu s$]')
    ax1.set_ylabel('electric signal [V]')
    ax1.legend(loc='best')

    map = ax2.imshow(img, cmap='vlag')
    fig.colorbar(map, ax=ax2)
    if pd.isna(img_file) == True:
        title = 'No picture taken.'
    elif fnmatch.fnmatch(img_file, '*SwY*'):
        title = 'Switchyard: '
        patch = patches.Ellipse((740,525),width=1050, height=1020, transform=ax2.transData, color='steelblue', linestyle='dotted', linewidth=1, alpha=1, fill=False)
        ax2.add_patch(patch)
    ax2.set_title(title)
    ax2.set_xlabel('y [pixels]')
    ax2.set_ylabel('z [pixels]')

    ax3.set_title('Silica PM - DRS')
    ax3.scatter(drs_time[peaks1], drs.ch1_corr[peaks1], color='firebrick', marker='x')
    ax3.plot(drs_time, drs.ch1_corr, label='Ch 1', color='indianred')
    ax3.scatter(drs_time[peaks2], drs.ch2_corr[peaks2],marker='x', color='orange')
    ax3.plot(drs_time, drs.ch2_corr, label='Ch 2', color='darkorange')
    ax3.scatter(drs_time[peaks3], drs.ch3_corr[peaks3],marker='x', color='forestgreen')
    ax3.plot(drs_time, drs.ch3_corr, label='Ch 3', color='limegreen')
    ax3.scatter(drs_time[peaks4], drs.ch4_corr[peaks4], marker='x', color='steelblue')
    ax3.plot(drs_time, drs.ch4_corr, label='Ch 4', color='dodgerblue')
    ax3.scatter(drs_time[peaks5], drs.ch5_corr[peaks5], marker='x', color='indigo')
    ax3.plot(drs_time, drs.ch5_corr, label='Ch 5', color='mediumorchid')
    
    ax3.set_xlabel('time [$\mu s$]')
    ax3.set_ylabel('electric signal')
    ax3.legend(loc='best')
    
    ax4.set_title('Electric signal MCP')
    ax4.scatter(wave_time_swy[peaks_swy], wave_swy.MCP_Swy[peaks_swy], color='steelblue', marker='x')
    ax4.plot(wave_time_swy, wave_swy, label='MCP Swy', color='dodgerblue')
    ax4.set_xlabel('time [$\mu s$]')
    ax4.set_ylabel('electric signal [V]')
    ax4.set_ylim(bottom=-0.1, top=0.35)
    [ax4.axvline(p, linestyle='-.', color='gray') for p in [delay_PCO, delay_PCO+1]]
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

    return 1

def checkDRS(timestamp, img_file, drs_file, clustering):
    timestamp = datetime.fromtimestamp(timestamp)
    img = getImg(img_file)

    drs_time, drs, peaks = getDRS(timestamp, drs_file)
    peaks1 = [int(item) for item in peaks.groupby('Time')['Peak1'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks2 = [int(item) for item in peaks.groupby('Time')['Peak2'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks3 = [int(item) for item in peaks.groupby('Time')['Peak3'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True] 
    peaks4 = [int(item) for item in peaks.groupby('Time')['Peak4'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    peaks5 = [int(item) for item in peaks.groupby('Time')['Peak5'].apply(list).reset_index(drop=True)[0] if not(math.isnan(item)) == True]
    
    fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2, figsize=(10,10))
    fig.suptitle(timestamp)

    ax1.set_title('Ch1 vs Ch2')
    ax1.scatter(drs_time[peaks1], drs.ch1_corr[peaks1], color='firebrick', marker='x')
    ax1.plot(drs.ch1_corr, label='Ch 1', color='indianred')
    ax1.scatter(drs_time[peaks2], drs.ch2_corr[peaks2],marker='x', color='orange')
    ax1.plot(drs.ch2_corr, label='Ch 2', color='darkorange')
    ax1.set_xlabel('time [$\mu s$]')
    ax1.set_ylabel('electric signal')
    ax1.legend(loc='best')

    ax2.set_title('Ch3 vs Ch4')
    ax2.scatter(drs_time[peaks3], drs.ch3_corr[peaks3],marker='x', color='forestgreen')
    ax2.plot(drs_time, drs.ch3_corr, label='Ch 3', color='limegreen')
    ax2.scatter(drs_time[peaks4], drs.ch4_corr[peaks4], marker='x', color='steelblue')
    ax2.plot(drs_time, drs.ch4_corr, label='Ch 4', color='dodgerblue')
    ax2.set_xlabel('time [$\mu s$]')
    ax2.set_ylabel('electric signal')
    ax2.legend(loc='best')
    
    ax3.set_title('Ch1 & Ch3 & Ch5')
    ax3.scatter(drs_time[peaks1], drs.ch1_corr[peaks1], color='firebrick', marker='x')
    ax3.plot(drs_time, drs.ch1_corr, label='Ch 1', color='indianred')
    ax3.scatter(drs_time[peaks3], drs.ch3_corr[peaks3],marker='x', color='forestgreen')
    ax3.plot(drs_time, drs.ch3_corr, label='Ch 3', color='limegreen')
    ax3.scatter(drs_time[peaks5], drs.ch5_corr[peaks5], marker='x', color='indigo')
    ax3.plot(drs_time, drs.ch5_corr, label='Ch 5', color='mediumorchid')
    ax3.set_xlabel('time [$\mu s$]')
    ax3.set_ylabel('electric signal')
    ax3.legend(loc='best')

    map = ax4.imshow(img, cmap='viridis') #could also use coolwarm
    fig.colorbar(map, ax=ax4)
    if pd.isna(img_file) == True:
        title = 'No picture taken.'
    elif fnmatch.fnmatch(img_file, '*SwY*'):
        title = 'Switchyard: '
        patch = patches.Ellipse((740,525),width=1050, height=1020, transform=ax4.transData, color='steelblue', linestyle='dotted', linewidth=1, alpha=1, fill=False)
        ax4.add_patch(patch)
    if clustering == True:
        blobs = clusterFinder(img)
        for blob in blobs:
            y,x,r = blob
            c = plt.Circle((x,y), r, color='orangered', linewidth=1, fill=False)
            ax4.add_patch(c)
            title = 'Detected clusters: '+str(len(blobs))
    ax4.set_title(title)
    ax4.set_xlabel('y [pixels]')
    ax4.set_ylabel('z [pixels]')
    
    plt.tight_layout
    plt.show()

