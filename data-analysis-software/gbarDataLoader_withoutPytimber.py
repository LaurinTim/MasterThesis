from pathlib import Path
import pandas as pd
import fnmatch, re
import matplotlib.pyplot as plt
from datetime import datetime
from time import mktime

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

def loadLog(date2an):
    if ((DATASUMMARY / date2an / ('log_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('log_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('LOG file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('log_'+date2an+'.txt')), sep='\t')
    else:
        if not (PROTONGUNPC / date2an).exists() or not (PROTONGUNPC / date2an / (date2an+'.txt')).is_file():
            print('Wrong date formate or date is not uploaded yet.')     
        else:
            if not (DATASUMMARY/ date2an).exists():
                Path.mkdir(DATASUMMARY / date2an) 
            logfile = PROTONGUNPC / date2an / (date2an+'.txt')
            log = pd.read_csv(logfile, sep='\t', header=None)
            # el_ = einzel lense, st_ = steerer, qt_ = quadrupol triplet, swy = switchyard, rc = reaction chamber, tl = transfer line, pbl = pbar line (after swy)
            log = log.rename(columns={0:'Time', 1:'valv_pos', 2:'mcp_pos', 3:'i_beam', 4:'wf',
                                    5:'amp_FC', 6:'amp_RC', 7:'amp_SWY', 8:'sum_pco'})
            log = log.rename(columns={9:'I_focus', 10:'uhf_read', 11:'is_pressure', 12:'u_beam',
                                    13:'u_focus', 14:'input1', 15:'input2'})
            log = log.rename(columns={16:'u_beam_set', 17:'u_focus_set', 18:'uhf_set', 19:'gas_inlet_set'})
            log = log.rename(columns={20:'el_pg1', 21:'el_pg2', 22:'el_pg3', 23:'wf_pos',
                                    24:'wf_neg', 25:'qb_pos', 26:'qb_neg', 27:'chop_neg'}) # proton line elements, monitored voltage [V]
            log = log.rename(columns={28:'el_tl1', 29:'el_tl2',
                                    30:'st_dec_up', 31:'st_dec_dw', 32:'st_dec_le', 33:'st_dec_ri',
                                    34:'st_tl_up', 35:'st_tl_dw', 36:'st_tl_le', 37:'st_tl_ri'})  # transfer line elements, monitored voltage [V]
            log = log.rename(columns={38:'el_ion1', 39:'el_ion2', 40:'el_ion3', 
                                    41:'st_ion_up', 42:'st_ion_dw', 43:'st_ion_le', 44:'st_ion_ri'}) # pbar line after swy elements, monitored voltage [V]
            log = log.rename(columns={45:'el_rc', 46:'st_rc1_up', 47:'st_rc1_dw', 48:'st_rc1_le', 49:'st_rc1_ri',
                                    50:'st_rc2_up', 51:'st_rc2_dw', 52:'st_rc2_le', 53:'st_rc2_ri',
                                    54:'st_rc3_up', 55:'st_rc3_dw', 56:'st_rc3_le', 57:'st_rc3_ri'}) # elements after the quad bender and before the MW transmission line
            log = log.rename(columns={58:'qt_1pos', 59:'qt_1neg', 60:'qt_2pos', 61:'qt_2neg', 62:'qt_3pos', 63:'qt_3neg'}) # quadrupol triplet [V]
            log = log.rename(columns={64:'swy1', 65:'swy2', 66:'swy3', 67:'swy4', 68:'el_swy'}) # monitored swy electrodes [V]
            log = log.rename(columns={69:'phos_ion', 70:'mcp_ion', 71:'phos_rc', 72:'mcp_rc', 73:'phos_swy', 74:'mcp_swy'}) # monitored mcp voltages [V]
            log = log.rename(columns={75:'defl', 76:'qnch_pos', 77:'qnch_neg'}) # single elements: deflector, quenching electrodes positive + negative, monitored [V], target height
            log = log.rename(columns={83:'pr_lya', 82:'pr_swy', 81:'pr_mw', 80:'pr_rc', 79:'pr_qb', 78:'pr_h'}) # monitored pressure in the vacuum system [mbar
            log = log.rename(columns={84:'H_offset', 85:'target', 86:'delay_A', 87:'delay_B', 88:'delay_C', 89:'delay_E', 90:'delay_G'})
            
            log.to_csv(str(DATASUMMARY / date2an / ('log_'+date2an+'.txt')), sep='\t', index=False)
            return log
        
mcp1_file = []
mcp1_time = []
mcp2_file = []
mcp2_time = []
mcp3_file = []
mcp3_time = []
mcp5_file = []
mcp5_time = []
mcp7_file = []
mcp7_time = []
def loadRCPC(date2an):
    if (RCPC / date2an).exists() == False:
        print('Files might not have been uploaded yet.')
    else:
        p = RCPC / date2an
        for file in p.iterdir():
            if fnmatch.fnmatch(file, '*MCP1*') and fnmatch.fnmatch(file, '*.tif'):
                mcp1_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp1_file.append(file)
            if fnmatch.fnmatch(file, '*MCP2*') and fnmatch.fnmatch(file, '*.tif'):
                mcp2_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp2_file.append(file)
            if fnmatch.fnmatch(file, '*MCP3*') and fnmatch.fnmatch(file, '*.tif'):
                mcp3_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp3_file.append(file)
            if fnmatch.fnmatch(file, '*PCO-ReC*') or fnmatch.fnmatch(file, '*MCP5*'):
                mcp5_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp5_file.append(file)
            if fnmatch.fnmatch(file, '*VCXG-51M-7*') and fnmatch.fnmatch(file, '*.tif'):
                mcp7_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp7_file.append(file)
        
    df1 = pd.DataFrame({'Time' : mcp1_time, 'MCP1' : mcp1_file})
    df2 = pd.DataFrame({'Time' : mcp2_time, 'MCP2' : mcp2_file})
    df3 = pd.DataFrame({'Time' : mcp3_time, 'MCP3' : mcp3_file})
    df5 = pd.DataFrame({'Time' : mcp5_time, 'MCP5' : mcp5_file})
    df7 = pd.DataFrame({'Time' : mcp7_time, 'MCP7' : mcp7_file})   
    return pd.concat([df1,df2,df3,df5,df7], ignore_index=True).sort_values('Time', ascending=True, ignore_index=True)
        
mcp6_time = []
mcp6_file = []
waveform_time = []
waveform_file = []
cmos_time = []
cmos_file = []
drs_time = []
drs_file = []
def loadPGUNPC(date2an):
    if (PROTONGUNPC / date2an).exists() == False:
        print('Files might not have been uploaded yet.')
    else:
        p = PROTONGUNPC / date2an
        for file in p.iterdir():
            if fnmatch.fnmatch(file, '*PCO-SwY*') or fnmatch.fnmatch(file, '*MCP6*'):
                mcp6_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp6_file.append(file)
            if fnmatch.fnmatch(file, '*WF*') and fnmatch.fnmatch(file,'*.txt'):
                waveform_time.append(int(re.search('WF\.(.*?)\.', str(file)).group(1)))
                waveform_file.append(file)
            if fnmatch.fnmatch(file, '*VCXG-51M-5*') and fnmatch.fnmatch(file, '*.tif'):
                cmos_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                cmos_file.append(file)
            if fnmatch.fnmatch(file, '*drs4*') and fnmatch.fnmatch(file, '*.txt'):
                filetime = re.search('drs4\_(.*?)\.', str(file)).group(1)
                drs_time.append(int(mktime(datetime(int('20'+filetime[0:2]),int(filetime[3:5]),int(filetime[6:8]),int(filetime[9:11]), int(filetime[12:14]), int(filetime[15:])).timetuple())))
                drs_file.append(file)

    df6 = pd.DataFrame({'Time' : mcp6_time, 'MCP6' : mcp6_file})
    dfwf = pd.DataFrame({'Time' : waveform_time, 'Waveform_12bit' : waveform_file})
    dfcmos = pd.DataFrame({'Time' : cmos_time, 'CMOS_Tracker' : cmos_file})
    dfdrs = pd.DataFrame({'Time' : drs_time, 'DRS4'  : drs_file})
    return pd.concat([df6, dfwf, dfcmos, dfdrs], ignore_index=True).sort_values('Time', ascending=True, ignore_index=True)

def loadELENA(date2an):
    if ((ELENADATA / date2an / ('elena_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((ELENADATA / date2an / ('elena_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('ELENA data for '+date2an+' already exists.')
        return pd.read_csv(str(ELENADATA / date2an / ('elena_'+date2an+'.txt')), sep='\t')
    else:
        print('ELENA data not yet loaded. Ask someone who has acess to NXCALS to load the datafile.')
 
        time = []
        intensity = []
        sem5020_amp_hor = []
        sem5020_amp_ver = []
        sem5060_amp_hor = []
        sem5060_amp_ver = []
        sem5020_mean_hor = []
        sem5020_mean_ver = []
        sem5060_mean_hor = []
        sem5060_mean_ver = []
        sem5020_sigma_hor = []
        sem5020_sigma_ver = []
        sem5060_sigma_hor = []
        sem5060_sigma_ver = []

        elena = pd.DataFrame({'Timestamp' : time, 'ELENA_Intensity' : intensity,
                            'SEM5020_gaussAmp_horizontal' : sem5020_amp_hor, 'SEM5020_gaussAmp_vertical' : sem5020_amp_ver,
                            'SEM5020_gaussMean_horizontal' : sem5020_mean_hor, 'SEM5020_gaussMean_vertical' : sem5020_mean_ver,
                            'SEM5020_gaussSig_horizontal' : sem5020_sigma_hor, 'SEM5020_gaussSig_vertical' : sem5020_sigma_ver,
                            'SEM5060_gaussAmp_horizontal' : sem5060_amp_hor, 'SEM5060_gaussAmp_vertical' : sem5060_amp_ver,
                            'SEM5060_gaussMean_horizontal' : sem5060_mean_hor, 'SEM5060_gaussMean_vertical' : sem5060_mean_ver,
                            'SEM5060_gaussSig_horizontal' : sem5060_sigma_hor, 'SEM5060_gaussSig_vertical' : sem5060_sigma_ver,}).sort_values('Timestamp', ascending=True, ignore_index=True).round({'Timestamp':0})
        return elena

def loadDatafile(date2an):
    if ((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d') ):
        print('SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('summary_'+date2an+'.txt')), sep='\t')
    else:
        print('Datafile not loaded yet. Please ask someone with access to NXCALS to load it.')
    return 0

def loadShortSummary(date2an):  
    if ((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('SHORT SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')), sep='\t')
    else:
        print('Datafile not loaded yet. Please ask someone with access to NXCALS to load it.')
    return 0
