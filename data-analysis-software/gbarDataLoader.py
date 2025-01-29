from pathlib import Path
import pandas as pd
import fnmatch, re
import pytimber
import matplotlib.pyplot as plt
from datetime import datetime
from time import mktime
from tqdm import tqdm
import lmfit
from readTrc_4CH import Trc

PROTONGUNPC = Path('/eos') / 'experiment' / 'gbar' /'pgunpc' / 'data'
RCPC = Path('/eos') / 'experiment' / 'gbar' / 'rcpc' / 'data'
ELENADATA = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'data24' / 'elenadata24' 
#This is where all the files get saved, currently my own directory since I cant edit or create files in the normal GBAR one
DATASUMMARY = Path('/eos') / 'user' / 'l' / 'lkoller' / 'GBAR' / 'datasummary'

#list with the names of the columns for the files in the pgungc/data/date/date.txt (replace date)
seq = ['Time', 'valv_pos', 'mcp_pos', 'i_beam', 'wf', 'amp_FC', 'amp_RC', 'amp_SWY', 'sum_pco', 'I_focus', 'uhf_read', 'is_pressure', 'u_beam', 'u_focus', 'input1', 'input2', 'u_beam_set', 'u_focus_set', 'uhf_set', 'gas_inlet_set', 'EL_pT_in', 'EL_pT_1', 'EL_pT_2', 'EL_pT_3', 'EL_pT_4', 'EL_pT_out', 'EL_TL1', 'EL_TL2', 'STdec_up', 'STdec_dw', 'STdec_ri', 'STdec_le', 'St_TL_up', 'St_TL_dw', 'St_TL_ri', 'St_TL_le', 'EL_pbL1', 'EL_pbL2', 'EL_pbl3', 'St_pbl_up', 'St_pbl_dw', 'St_pbl_le', 'St_pbl_ri', 'pbT_in_+', 'pbT_out_+', 'Ly_MCP_1', 'Ly_MCP_2', 'Ly_MCP_3', 'Ly_MCP_4', 'Sci_1', 'Sci_2', 'EL_RC1', 'St_RC1_up', 'St_RC1_dw', 'St_RC1_ri', 'St_RC1_le', 'St_RC2_up', 'St_RC2_dw', 'St_RC2_le', 'St_RC2_ri', 'St_RC3_up', 'St_RC3_dw', 'St_RC3_le', 'St_RC3_ri', 'QT_RC1_+', 'QT_RC1_-', 'QT_RC12+', 'QT_RC2_-', 'QT_RC3_+', 'QT_RC3_-', 'SwY_1_+', 'SwY_2_-', 'SwY_3_+', 'SwY_4_-', 'SWY_EL', '6_PbT_pho_+', '6_PbT_mcp_+', '4_RC_pho_+', '4_RC_mcp_+', '5_SwY_pho_+', '5_SwyY_mcp_+', 'Defl_+', 'Qnch_+', 'Qnch_-', 'pr_h', 'pr_qb', 'pr_rc', 'pr_mw', 'pr_swy', 'pr_lya','H_offset', 'target', 'delay_A', 'delay_B', 'delay_C', 'delay_E', 'delay_G', 'MCP_front_bias']

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

def loadLog(date2an):
    '''
    Load Christians DAQ file from the protongun PC and label the columns with the seq list
    
    Parameters
    ------------
    date2an = the date for which we want to create a log file
    
    '''
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
            
            for i in range(len(seq)):
                log = log.rename(columns={i: seq[i]})
            
            '''
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
            log = log.rename(columns={84:'H_offset', 85:'target', 86:'delay_A', 87:'delay_B', 88:'delay_C', 89:'delay_E', 90:'delay_G'})'''
            
            log.to_csv(str(DATASUMMARY / date2an / ('log_'+date2an+'.txt')), sep='\t', index=False)
            return log
        


mcp1_file = []
mcp1_time = []
mcp2_file = []
mcp2_time = []
mcp3_file = []
mcp3_time = []
mcp4_file = []
mcp4_time = []
mcp7_file = []
mcp7_time = []
def loadRCPC(date2an):
    '''
    Get the times and locations for the files from the different MCPs which are in the RCPC folder
    
    There may be new MCPs installed since 2022
    
    Parameters
    ------------
    date2an = the date for which we want to create a log file
    
    Returns
    ------------
    dataframe with the times and filenames for the MCPSs at date2an sorted by time
    
    '''
    if (RCPC / date2an).exists() == False:
        print('RCPC files might not have been uploaded yet.')
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
                mcp4_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp4_file.append(file)
            if fnmatch.fnmatch(file, '*VCXG-51M-7*') and fnmatch.fnmatch(file, '*.tif'):
                mcp7_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                mcp7_file.append(file)
        
    df1 = pd.DataFrame({'Time' : mcp1_time, 'MCP1' : mcp1_file})
    df2 = pd.DataFrame({'Time' : mcp2_time, 'MCP2' : mcp2_file})
    df3 = pd.DataFrame({'Time' : mcp3_time, 'MCP3' : mcp3_file})
    df4 = pd.DataFrame({'Time' : mcp4_time, 'MCP4' : mcp4_file})
    df7 = pd.DataFrame({'Time' : mcp7_time, 'MCP7' : mcp7_file})  
    return pd.concat([df1,df2,df3,df4,df7], ignore_index=True).sort_values('Time', ascending=True, ignore_index=True)
        
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
    dataframe with all the data from above sorted by time
    
    '''
    if (PROTONGUNPC / date2an).exists() == False:
        print('PGunPC files might not have been uploaded yet.')
    else:
        p = PROTONGUNPC / date2an
        posi = p / (date2an+'posi')
        lya = p / (date2an+'lya')
        for file in p.iterdir():
            if fnmatch.fnmatch(file, '*PCO-SwY*') or fnmatch.fnmatch(file, '*MCP6*'):
                mcp5_time.append(int(re.search('s\_(.*?)\.', str(file)).group(1)))
                mcp5_file.append(file)
            if fnmatch.fnmatch(file, '*WF1234*') and fnmatch.fnmatch(file,'*.trc'):
                waveform_time.append(int(re.search('WF1234\.(.*?)\.', str(file)).group(1)))
                waveform_file.append(file)
            if fnmatch.fnmatch(file, '*VCXG-51M-5*') and fnmatch.fnmatch(file, '*.tif'):
                cmos_time.append(int(re.search('G16\_(.*?)\.', str(file)).group(1)))
                cmos_file.append(file)
            if fnmatch.fnmatch(file, '*drs4*') and fnmatch.fnmatch(file, '*.txt'):
                filetime = re.search('drs4\_(.*?)\.', str(file)).group(1)
                drs_time.append(int(mktime(datetime(int('20'+filetime[0:2]),int(filetime[3:5]),int(filetime[6:8]),int(filetime[9:11]), int(filetime[12:14]), int(filetime[15:17])).timetuple())))
                drs_file.append(file)
        if (posi.exists()):
            for file in posi.iterdir():
                if fnmatch.fnmatch(file, '*C1*') and fnmatch.fnmatch(file, '*.trc'):
                    posi_c1_time.append(int(re.search('C1\.(.*?)\.', str(file)).group(1)))
                    posi_c1_file.append(file)
                if fnmatch.fnmatch(file, '*C2*') and fnmatch.fnmatch(file, '*.trc'):
                    posi_c2_time.append(int(re.search('C2\.(.*?)\.', str(file)).group(1)))
                    posi_c2_file.append(file) 
                if fnmatch.fnmatch(file, '*C3*') and fnmatch.fnmatch(file, '*.trc'):
                    posi_c3_time.append(int(re.search('C3\.(.*?)\.', str(file)).group(1)))
                    posi_c3_file.append(file)
                if fnmatch.fnmatch(file, '*C4*') and fnmatch.fnmatch(file, '*.trc'):
                    posi_c4_time.append(int(re.search('C4\.(.*?)\.', str(file)).group(1)))
                    posi_c4_file.append(file)
                if fnmatch.fnmatch(file, '*SD*') and fnmatch.fnmatch(file, '*.png'):
                    sd_posi_time.append(int(re.search('SD\.(.*?)\.', str(file)).group(1)))
                    sd_posi_file.append(file)
        if (lya.exists()):
            for file in lya.iterdir():
                if fnmatch.fnmatch(file, '*LY*') and fnmatch.fnmatch(file, '*.trc'):
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
                if fnmatch.fnmatch(file, '*SD*') and fnmatch.fnmatch(file, '*.png'):
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

def loadELENA(date2an):
    '''
    
    '''
    #if ((ELENADATA / date2an / ('elena_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((ELENADATA / date2an / ('elena_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
    #    print('ELENA data for '+date2an+' already exists.')
    #    return pd.read_csv(str(ELENADATA / date2an / ('elena_'+date2an+'.txt')), sep='\t')
    #else:
    ldb = pytimber.LoggingDB(spark_session = spark)
    if not (ELENADATA / date2an).exists():
        Path.mkdir(ELENADATA / date2an) 

    t1 = '20'+date2an[0:2]+'-'+date2an[3:5]+'-'+date2an[6:8]+' 00:00:00.000'
    t2 = '20'+date2an[0:2]+'-'+date2an[3:5]+'-'+date2an[6:8]+' 23:59:59.000'

    # GET the data from NXCALS via PyTimber
    ELENA_intensity = ldb.get(['LNE.APULB.5030:INTENSITY', 'LNE.APULB.0030:Acquisition:totalIntensitySingle'], t1,t2)
    SEM_amplitude = ldb.get(['LNE50.BSGW.5020:FitAcq:gaussAmplitude','LNE50.BSGW.5060:FitAcq:gaussAmplitude'], t1,t2)
    SEM_mean = ldb.get(['LNE50.BSGW.5020:FitAcq:gaussMean', 'LNE50.BSGW.5060:FitAcq:gaussMean'], t1,t2)
    SEM_sigma = ldb.get(['LNE50.BSGW.5020:FitAcq:gaussSigma', 'LNE50.BSGW.5060:FitAcq:gaussSigma'], t1,t2)
    SEM_in = ldb.get(['LNE50.BSGW.5020:Acquisition:isMonitorIn', 'LNE50.BSGW.5060:Acquisition:isMonitorIn'], t1,t2)
    intensity_1 = []
    intensity_2 = []
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
    sem5020_in = []
    sem5060_in = []

    time1 = [ELENA_intensity['LNE.APULB.5030:INTENSITY'][0][i] for i in range(len(ELENA_intensity['LNE.APULB.5030:INTENSITY'][0]))]
    time2 = [SEM_amplitude['LNE50.BSGW.5020:FitAcq:gaussAmplitude'][0][i] for i in range(len(SEM_amplitude['LNE50.BSGW.5020:FitAcq:gaussAmplitude'][0]))]

    diff = list(set(time1)-set(time2))

    for idx1, timestamp1 in enumerate(ELENA_intensity['LNE.APULB.5030:INTENSITY'][0]):
        for dt in diff:
            if timestamp1==dt:
                #time1.append(timestamp1)
                intensity_1.append(ELENA_intensity['LNE.APULB.5030:INTENSITY'][1][idx1][0])
                intensity_2.append(ELENA_intensity['LNE.APULB.0030:Acquisition:totalIntensitySingle'][1][idx1][1])
                sem5020_amp_hor.append(0)
                sem5020_amp_ver.append(0)
                sem5060_amp_hor.append(0)
                sem5060_amp_ver.append(0)
                sem5020_mean_hor.append(0)
                sem5020_mean_ver.append(0)
                sem5060_mean_hor.append(0)
                sem5060_mean_ver.append(0)
                sem5020_sigma_hor.append(0)
                sem5020_sigma_ver.append(0)
                sem5060_sigma_hor.append(0)
                sem5060_sigma_ver.append(0)
                sem5020_in.append(0)
                sem5060_in.append(0)
        for idx2,timestamp2 in enumerate(SEM_amplitude['LNE50.BSGW.5020:FitAcq:gaussAmplitude'][0]):
            if timestamp1==timestamp2:
                #time1.append(timestamp1)
                intensity_1.append(ELENA_intensity['LNE.APULB.5030:INTENSITY'][1][idx1][0])
                intensity_2.append(ELENA_intensity['LNE.APULB.0030:Acquisition:totalIntensitySingle'][1][idx1][1])
                sem5020_amp_hor.append(SEM_amplitude['LNE50.BSGW.5020:FitAcq:gaussAmplitude'][1][idx2][0])
                sem5020_amp_ver.append(SEM_amplitude['LNE50.BSGW.5020:FitAcq:gaussAmplitude'][1][idx2][1])
                sem5060_amp_hor.append(SEM_amplitude['LNE50.BSGW.5060:FitAcq:gaussAmplitude'][1][idx2][0])
                sem5060_amp_ver.append(SEM_amplitude['LNE50.BSGW.5060:FitAcq:gaussAmplitude'][1][idx2][1])
                sem5020_mean_hor.append(SEM_mean['LNE50.BSGW.5020:FitAcq:gaussMean'][1][idx2][0])
                sem5020_mean_ver.append(SEM_mean['LNE50.BSGW.5020:FitAcq:gaussMean'][1][idx2][1])
                sem5060_mean_hor.append(SEM_mean['LNE50.BSGW.5060:FitAcq:gaussMean'][1][idx2][0])
                sem5060_mean_ver.append(SEM_mean['LNE50.BSGW.5060:FitAcq:gaussMean'][1][idx2][1])
                sem5020_sigma_hor.append(SEM_sigma['LNE50.BSGW.5020:FitAcq:gaussSigma'][1][idx2][0])
                sem5020_sigma_ver.append(SEM_sigma['LNE50.BSGW.5020:FitAcq:gaussSigma'][1][idx2][1])
                sem5060_sigma_hor.append(SEM_sigma['LNE50.BSGW.5060:FitAcq:gaussSigma'][1][idx2][0])
                sem5060_sigma_ver.append(SEM_sigma['LNE50.BSGW.5060:FitAcq:gaussSigma'][1][idx2][1])
                sem5020_in.append(SEM_in['LNE50.BSGW.5020:Acquisition:isMonitorIn'][1][idx2])
                sem5060_in.append(SEM_in['LNE50.BSGW.5060:Acquisition:isMonitorIn'][1][idx2])

    elena = pd.DataFrame({'Timestamp' : time1, 'ELENA_Intensity' : intensity_1, 'TotalIntensity' : intensity_2,
                        'SEM5020_gaussAmp_horizontal' : sem5020_amp_hor, 'SEM5020_gaussAmp_vertical' : sem5020_amp_ver,
                        'SEM5020_gaussMean_horizontal' : sem5020_mean_hor, 'SEM5020_gaussMean_vertical' : sem5020_mean_ver,
                        'SEM5020_gaussSig_horizontal' : sem5020_sigma_hor, 'SEM5020_gaussSig_vertical' : sem5020_sigma_ver,
                        'SEM5060_gaussAmp_horizontal' : sem5060_amp_hor, 'SEM5060_gaussAmp_vertical' : sem5060_amp_ver,
                        'SEM5060_gaussMean_horizontal' : sem5060_mean_hor, 'SEM5060_gaussMean_vertical' : sem5060_mean_ver,
                        'SEM5060_gaussSig_horizontal' : sem5060_sigma_hor, 'SEM5060_gaussSig_vertical' : sem5060_sigma_ver,
                        'SEM5020_in': sem5020_in, 'SEM5060_in': sem5060_in}).sort_values('Timestamp', ascending=True, ignore_index=True).round({'Timestamp':0})
    elena.to_csv(str(ELENADATA / date2an / ('elena_'+date2an+'.txt')), sep='\t', index=False)
    return elena

def loadDatafile(date2an):
    if ((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).is_file() == True) and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('summary_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d') ):
        print('SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('summary_'+date2an+'.txt')), sep='\t')
    else:
        if not (DATASUMMARY/ date2an).exists():
            Path.mkdir(DATASUMMARY / date2an) 
        dfpgun = loadPGUNPC(date2an)
        #dfelena = loadELENA(date2an)
        dfrcpc = loadRCPC(date2an)
        dflog = loadLog(date2an)

        #ldb = pytimber.LoggingDB(source='nxcals')
        #day = int(date2an[6:8])
            
        #t1 = '20'+date2an[0:2]+'-'+date2an[3:5]+'-'+str(day-1)+' 23:00:00.000'
        #t2 = '20'+date2an[0:2]+'-'+date2an[3:5]+'-'+str(day+1)+' 01:59:59.000'
        #print(day, t1,t2)
        #BEAM_stopper = ldb.get('LNE.TBS.5040:ACQ_POSITION', t1,t2)
        #stopper_time, stopper_value = [],[]
        #for idx, timestamp in enumerate(BEAM_stopper['LNE.TBS.5040:ACQ_POSITION'][0]):
        #    stopper_time.append(timestamp)
        #    stopper_value.append(BEAM_stopper['LNE.TBS.5040:ACQ_POSITION'][1][idx])
        #df_stopper = pd.DataFrame({'Timestamp':stopper_time, 'BEAM_Stopper':stopper_value})

        datafiles= pd.concat([dfpgun, dfrcpc, dflog], ignore_index=True).round({'Time':0})
    
        sequence = seq
        
        #for idx, timestamp in tqdm(enumerate(dfelena['Timestamp'])):
        for idx, timestamp in tqdm(enumerate(dflog['Time'])):
            for label in sequence:
                for idy, file in enumerate(datafiles[label]):
                    if ((timestamp+20) > datafiles['Time'][idy] >= timestamp-20) and pd.isna(file)==False:
                        #print(timestamp, " ", idx, " ", datafiles['Time'][idy], " ", label, " ", file)
                        #dfelena.loc[idx, label] = file
                        dflog.loc[idx,label] = file
            #for label in sequence_logFile:
            #    for idy, value in enumerate(datafiles[label]):
            #        if ((timestamp+25) > datafiles['Time'][idy] >= timestamp) and pd.isna(value)==False:
                        #print(timestamp, " ", idx, " ", datafiles['Time'][idy], " ", label, " ", value)
            #            dfelena.loc[idx, label] = value

        df_final = pd.DataFrame()
        df_mid = pd.DataFrame()
        df_mid_stop = pd.DataFrame()
        #for idx in range(df_stopper.shape[0]-1):
        #    if idx==0:
        #        df = dfelena[(dfelena['Timestamp'] < df_stopper['Timestamp'][idx])].reset_index(drop=True)
        #        size = df.shape[0]
        #        df_mid_stop = pd.DataFrame({'BEAM_Stopper':[df_stopper['BEAM_Stopper'][idx] for i in range(size)]})
        #        df_mid = pd.concat([df,df_mid_stop],axis=1)
        #        df_final = pd.concat([df_final,df_mid],axis=0).reset_index(drop=True)

        #    df = dfelena[(dfelena['Timestamp'] > df_stopper['Timestamp'][idx]) & (dfelena['Timestamp'] <= df_stopper['Timestamp'][idx+1])].reset_index(drop=True)
        #    size = df.shape[0]
        #    df_mid_stop = pd.DataFrame({'BEAM_Stopper':[df_stopper['BEAM_Stopper'][idx] for i in range(size)]})
        #    df_mid = pd.concat([df,df_mid_stop],axis=1)
        #    df_final = pd.concat([df_final,df_mid], axis=0).reset_index(drop=True)
        dflog['Timestamp'] = dflog['Time']
        df_final = dflog

        df_final = pd.concat([df_final, pd.DataFrame({'Datetime' : [datetime.fromtimestamp(entry) for entry in df_final['Timestamp']]})], axis=1)
        df_final.to_csv(str(DATASUMMARY / date2an / ('summary_'+date2an+'.txt')), sep='\t', index=False)
    return df_final

def loadShortSummary(date2an):  
    if ((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).is_file() == True):# and (date2an < datetime.fromtimestamp((DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')).stat().st_mtime).strftime('%y_%m_%d')):
        print('SHORT SUMMARY file for '+date2an+' already exists.')
        return pd.read_csv(str(DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')), sep='\t')
    else: 
        longData = loadDatafile(date2an)
        #elena = loadELENA(date2an)
        log = loadLog(date2an)

        #shortData = longData.dropna(thresh=(elena.shape[1]+log.shape[1]+3))
        shortData = longData.dropna(thresh=(log.shape[1]+3))
        shortData.to_csv(str(DATASUMMARY / date2an / ('shortSum_'+date2an+'.txt')), sep='\t', index=False)
    return shortData