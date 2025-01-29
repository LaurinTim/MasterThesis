#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff')
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
from time import time
import scipy.signal as sci
import scipy.optimize as scio
from tqdm import tqdm
from volt_corr_funcs import corr_func


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

'''
#With corr for datasets 1 and 2
def corr_func0(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[-0.0015137950944264093, -0.02744247831017351, -0.16562848248495723, -0.3330093029569526],
[0.03951024089276757, 0.6200004485062616, 3.2342471474823773, 5.609106493016304],
[-0.01632374750256321, -0.053263549562328726, 0.49230302776857576, 1.83651774117784],
[-0.01323332773041649, -0.14818771841165868, -0.1182333098647983, 0.94102951385338],
[-0.015952568015033305, -0.14221649523951266, -0.07915096634058365, 0.9743463630698183],
[-0.0032507940714410904, -0.07970021534468416, -0.0020329939977791135, 1.0026515623162469],
[0.005516756941610901, -0.04853593377916741, -0.011517392777784251, 1.0030735218505125],
[0.0016728375426850548, -0.014157078248380469, -0.0789179705747563, 1.0414571241968225],
[-0.01758465828143181, 0.1293340571191346, -0.4582877886148147, 1.391260950767553],
[0.019615438965390773, -0.2886681127069734, 1.0936184772747497, -0.5166907084779129],
[0.05060784288582057, -0.7385854028188654, 3.279779538501008, -4.068943910586589],
[-0.04074157086548274, 0.8229256439021995, -5.640191848784, 12.957446413749887]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func1(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[0.030772028955111035, 0.5522049613605023, 3.293412926985088, 6.525407333361371],
[0.02150461326936306, 0.36986010844744416, 2.1480616988846246, 4.196304186691701],
[-0.015089360731411969, -0.12117347748126103, -0.0037175657452006624, 1.1048199453086942],
[-0.030486221584615926, -0.2632626518468719, -0.4277423621119762, 0.6961124752914659],
[-0.031171113272144963, -0.20020168618333728, -0.18907614668507494, 0.9160742151926339],
[-0.006095455782042671, -0.0637965969691081, -0.002462836614618356, 0.9911313361808016],
[0.007449270830959001, -0.03451331093975664, -0.008364896726708562, 0.9910974956181385],
[-0.0010818935067141578, -0.0038572584845160663, -0.053408105875245526, 1.0165851145648854],
[0.005475212041614712, -0.1098256429175186, 0.3350893594503829, 0.6023152654953464],
[0.042441665962976695, -0.5597882434825345, 2.1480322745767126, -1.8160008677502595],
[-0.010973910029604093, 0.20371475968622568, -1.4887514723061257, 3.957613754877198],
[-0.015222776956155919, 0.3442993231755323, -2.664916883436106, 6.882249279641919]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func2(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[-0.01119577886653246, -0.19369401308784548, -1.1051795384910181, -2.0746623136193296],
[0.02618848219689518, 0.3807089250164999, 1.842104849621729, 2.9761016454739937],
[0.05125963250584738, 0.6789408275930028, 2.9962934340756133, 4.42411750949024],
[-0.043570990479047104, -0.2701841123536834, -0.13665814885467556, 1.0033854236040902],
[-0.019744045565520726, -0.21884753097609616, -0.15058592630820777, 0.9559038907395463],
[-0.015740524347509614, -0.10161217830795466, -0.0005809170682841946, 1.0001288871575302],
[0.01231790645464614, -0.06995431294000745, -0.011794343689378348, 1.0007614391662059],
[0.013203978576528242, -0.07767858167559737, 0.0009852707728576612, 0.9951989598888816],
[0.003069642906659032, -0.012532918808393204, -0.150253681667138, 1.122887529139279],
[0.005483930663570783, -0.10235384451090157, 0.3624958944124141, 0.3221550286848394],
[0.0319989714162176, -0.4852941958874221, 2.216367769263675, -2.6830783833086524],
[-0.011507093079841263, 0.25953213753226556, -2.0408089004060534, 5.44061939436045]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func3(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[0.01830205810744437, 0.3275152740314739, 1.9475520045284278, 3.845090155927785],
[0.025611095313990417, 0.42703056945277706, 2.3782535262537485, 4.42190814374269],
[-0.009465529763028653, -0.048304717354021955, 0.2616002821439743, 1.3149846627448674],
[-0.014239257740263335, -0.10647433522615869, 0.0520554949733842, 1.0811740929696387],
[-0.0161499990708698, -0.13552388960437498, -0.034985709671962656, 1.007864664391964],
[-0.005699455300548043, -0.10126625587236313, -0.0006882322891283593, 1.0184807103364315],
[0.004669413499323022, -0.08902509528624633, -0.00494703135006499, 1.0187941513468604],
[0.014320975121177462, -0.08909559727674315, -0.04471633349199468, 1.050169564725147],
[0.0013569133691834364, -0.0007522032163031058, -0.2564330072899866, 1.22824641292024],
[0.013664700854334773, -0.14913985841754696, 0.32931639220633097, 0.4675344068982346],
[-0.011098767046817866, 0.19696631297817777, -1.288999892473908, 2.998573707391776],
[0.025704282118190643, -0.4546168170260647, 2.556169345299582, -4.5645063827629215]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret'''


#With corr for dataset 3
def corr_func0(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[-0.0014433743660386036, -0.026778019136001932, -0.16518901465580477, -0.33903478844944346],
[0.041366358519031816, 0.6465372654760115, 3.3561059087229266, 5.786009308254869],
[-0.007141662352840599, 0.03568143274297661, 0.778591947991081, 2.142648514946095],
[-0.020601281787842166, -0.20772486887621783, -0.26831903294160164, 0.8286341951644727],
[-0.02156483135916729, -0.164540819824305, -0.1163023921420822, 0.9524885343119878],
[0.0018306999033595348, -0.06963852051668303, -0.0007825173411241274, 0.9968169624318685],
[0.007462873299133612, -0.055296218483558465, -0.005470814072686979, 0.9971626158259148],
[0.0007136952644362932, -0.002890024214821936, -0.10421413405641039, 1.052502561304142],
[-0.017302722799651157, 0.11754557982059675, -0.3909171174517562, 1.2952293270413076],
[0.03783128305478139, -0.5141359102903686, 2.014599937866297, -1.7511786743145488],
[0.0379987430659651, -0.5462982155695473, 2.306571005456471, -2.4295064748555295],
[-0.04028062867804404, 0.8129741820594287, -5.573679800500307, 12.8211379151236]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func1(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[0.024456747458263425, 0.44373130164944447, 2.673964836120816, 5.349164501647793],
[0.021209587957149414, 0.37517908357116675, 2.213752385952291, 4.353156798518895],
[-0.005786960123326071, -0.01597915609252152, 0.3905117807722026, 1.5912780464773932],
[-0.01907517579749326, -0.1844413024702908, -0.2450886009945044, 0.8420099275545401],
[-0.028055379780758147, -0.1966199450502089, -0.19983763510002273, 0.908888497561581],
[-0.0013659284458832628, -0.055075645576450234, -0.0012244095479740107, 0.99179890829867],
[0.008720857520066542, -0.039134693136241364, -0.006917817591917617, 0.9923542304059211],
[-0.008020660783951269, 0.038188684389708606, -0.13516431625730774, 1.0655169199011465],
[-0.0044318233165867625, -0.016619544467543032, 0.0517357169442525, 0.8817006242909634],
[0.05534145540886056, -0.7145373082375952, 2.7592304051605625, -2.608595876752945],
[-0.0002931207018829531, 0.038907597218231664, -0.6498903987838002, 2.5455990213393815],
[-0.011544799829774076, 0.2796205702118046, -2.287470289689842, 6.1437240572285114]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func2(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[-0.002952532349051725, -0.05011148240398756, -0.27807648192251827, -0.49952806814643846],
[0.03333143672873535, 0.4946876411048361, 2.441733226775185, 4.014372742875839],
[0.03751256034874038, 0.5160595403020707, 2.3809067232498977, 3.6939076195651563],
[-0.02775356669208224, -0.17156748005093314, 0.0496899443383914, 1.124370265281721],
[-0.03634109658006778, -0.2703524834303742, -0.20943480114011467, 0.9304065450004148],
[-0.006021475295509869, -0.09445688903517184, -0.00042569011106448963, 0.9966635899252401],
[0.01854162546533545, -0.08245789306511367, -0.00539292573871123, 0.9972535929640681],
[0.0115570377682143, -0.06355008737650976, -0.023205009524739813, 1.0037638735055858],
[-0.00503384654796403, 0.03831239979368845, -0.24416186787880342, 1.1760792314607422],
[0.015566103197245692, -0.2183461702043952, 0.7887251469550608, -0.17981465641582395],
[0.014790146810892996, -0.2140917549638169, 0.80537817346953, -0.2703136410443189],
[-0.00042011491945029815, 0.049416650955109005, -0.7182358986159416, 2.6697271617850498]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret

def corr_func3(x, t, h):
    time_steps = np.linspace(-6.282, 6.716999999999999, 13)
    
    args = [[0.019109669846021998, 0.34358025390524755, 2.0502782871584797, 4.058224373186713],
[0.01850271107163711, 0.3186382445423866, 1.840509186820943, 3.556209652201634],
[-0.006917986211623395, -0.016546101136060573, 0.3876798602170615, 1.4815052455154265],
[-0.0253583039865651, -0.19244927254266955, -0.16617622362544004, 0.9051584425987443],
[-0.022205710643891804, -0.1587772200657639, -0.07204032081964183, 0.983448644583166],
[-0.010955777210953335, -0.10559553796326177, -0.0018541501369063033, 1.0116841872548035],
[0.0056029160771875405, -0.08587062206153527, -0.003773240207848923, 1.0113610657386667],
[0.01062332304897469, -0.06766750855207367, -0.0776085084275031, 1.0662867109348853],
[0.00841026750139348, -0.0682215398790671, -0.046218324704207996, 1.0238728682240372],
[0.0005102703145978246, 0.011966759868564679, -0.32007805416220836, 1.338693315651659],
[-0.0006637175841516573, 0.03151357675206714, -0.42575138189503503, 1.5254492139247064],
[0.0067428301002163665, -0.09808744562668871, 0.3309120513534426, 0.05163349131801076]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = max(0,(args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h)
    
    return ret


def pf_try20(x, t1, t2, h1, h2): 
    y = [val+bal for val,bal in zip(corr_func0(x, t1, h1),corr_func0(x, t2, h2))]
    return y

def pf_try21(x, t1, t2, h1, h2): 
    y = [val+bal for val,bal in zip(corr_func1(x, t1, h1),corr_func1(x, t2, h2))]
    return y

def pf_try22(x, t1, t2, h1, h2): 
    y = [val+bal for val,bal in zip(corr_func2(x, t1, h1),corr_func2(x, t2, h2))]
    return y

def pf_try23(x, t1, t2, h1, h2): 
    y = [val+bal for val,bal in zip(corr_func3(x, t1, h1),corr_func3(x, t2, h2))]
    return y


#class for 1 trc file with data from the lyman alpha detectors
class Lfile():
    '''
    
    Class for a single trc file which gives info like the peak locations and heights and whether the mw was off/on for the measurement
    
    Parameters
    ------------
    path of the file
    
    '''
    
    def __init__(self, filepath, df = None, corr_df = None):
        self.filepath = filepath
        
        try:
            self.data = list(Trc().open(Path(self.filepath)))
        except:
            self.valid = False
            print('Trc readout error for file ', self.filepath)
        else:
            self.valid = True
            self.data[1] = [-self.data[1][0], -self.data[1][1], -self.data[1][2], -self.data[1][3]] #change the sign of the voltages (data[1][0-3]) so we can search for maxima peaks  
            self.data[0] = [np.linspace(0, len(self.data[0][0])-1, len(self.data[0][0])), np.linspace(0, len(self.data[0][1])-1, len(self.data[0][1])), np.linspace(0, len(self.data[0][2])-1, len(self.data[0][2])), np.linspace(0, len(self.data[0][3])-1, len(self.data[0][3]))]
            for i in range(4):
                av = np.average(self.data[1][i][8000:10000])
                self.data[1][i] = [val - av for val in self.data[1][i]]
        if type(df) != type(None):
            self.df = df
        else:
            self.df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
            
        if type(corr_df) != type(None):
            self.corr_df = corr_df.fillna(-100)
        else:
            self.corr_df = pd.read_csv('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/correction_arrays.csv').fillna(-100)
        
    def __str__(self):
        return f'{self.filepath}'
    
    
    def parpeak(self, ch, peak, edges = [0, 0]):
        '''
        
        get the actual peak and timing for a voltage pulse from the LyA trc files by fitting a parabola to the 3 elements before the highest measured voltage and 1 point after
        if the parabola has a lower peak than the voltage point, we keep the voltage as the pulse height

        Parameters
        ------------
        ch: channel in which the peak was recorded
        ts: timestep of the peak
        edges: left and right edges of the plateau for the pulse

        Returns
        ------------
        t: time of the peak of the parabola
        h: height of the peak of the parabola or max measured pulse voltage, whichever is higher (gives the peak as a positive number)
        
        '''
        dtp = 0
        dtcf = 0
        
        stp = time()
        
        xdat = self.data[0][0]
        ydat = self.data[1][ch]
        
        if edges[1]-edges[0] <= 2:
            xcurr = xdat[peak-5:peak+6]
            ycurr = ydat[peak-5:peak+6]

        else:
            xcurr = np.array(list(xdat[edges[0]-3:edges[0]]) + list(xdat[edges[1]+1:edges[1]+5]))
            ycurr = np.array(list(ydat[edges[0]-3:edges[0]]) + list(ydat[edges[1]+1:edges[1]+5]))
            
        if len(xcurr) == 0: xcurr = [-1]
        
        try:
            stcf = time()
            if ch == 0:   args = scio.curve_fit(corr_func0, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]], bounds = ([xcurr[0], 0.005], [xcurr[-1], 2]))
            elif ch == 1: args = scio.curve_fit(corr_func1, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]], bounds = ([xcurr[0], 0.005], [xcurr[-1], 2]))
            elif ch == 2: args = scio.curve_fit(corr_func2, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]], bounds = ([xcurr[0], 0.005], [xcurr[-1], 2]))
            elif ch == 3: args = scio.curve_fit(corr_func3, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]], bounds = ([xcurr[0], 0.005], [xcurr[-1], 2]))
            dtcf += time() - stcf
        except Exception as ex:
            t, h = peak, ydat[peak]
            diff = 1
        else: 
            t, h = args[0][0], args[0][1] 
            if ch == 0:   diff = sum([abs(val-bal)**2 for val,bal in zip(ycurr, corr_func0(xcurr, t, h))])#**0.5
            elif ch == 1: diff = sum([abs(val-bal)**2 for val,bal in zip(ycurr, corr_func1(xcurr, t, h))])#**0.5
            elif ch == 2: diff = sum([abs(val-bal)**2 for val,bal in zip(ycurr, corr_func2(xcurr, t, h))])#**0.5
            elif ch == 3: diff = sum([abs(val-bal)**2 for val,bal in zip(ycurr, corr_func3(xcurr, t, h))])#**0.5
        
        if h == 0: h = 0.001
        
        dtp += time() - stp
        
        return t, h, diff/(h*len(xcurr))#, dtp, dtcf
    
    
    def parpeak2(self, ch, peak):
        '''

        

        Parameters
        ------------
        ch: channel in which the peak was recorded
        ts: timestep of the peak

        Returns
        ------------
        args[0][0]: time of the first peak
        args[0][1]: time of the first peak
        args[0][2]: height of the first peak
        args[0][3]: height of the first peak

        '''
        dt2p = 0
        dt2cf1 = 0
        dt2cf2 = 0
        
        st2p = time()
        
        xdat = self.data[0][0]
        ydat = self.data[1][ch]
        
        lst1 = 0
        rst1 = 0

        while peak-lst1>=1 and lst1<20 and ydat[peak-lst1-1]>=0.003:
            lst1 += 1

        while peak+rst1<=10000 and rst1<20 and ydat[peak+rst1+1]>=0.003:
            rst1 += 1

        lst1 += 1
        rst1 += 2
        
        if lst1+rst1 <= 5: return [-1, -1, 5, 5], 10

        xcurr1 = xdat[peak-lst1:peak+rst1]
        ycurr1 = ydat[peak-lst1:peak+rst1]
        if len(xcurr1) == 0: xcurr1 = [-1]

        try:
            st2cf1 = time()
            if ch == 0: args1 = scio.curve_fit(pf_try20, xcurr1, ycurr1, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr1[0], xcurr1[0], 0.005, 0.005], [xcurr1[-1], xcurr1[-1], 2, 2]))
            if ch == 1: args1 = scio.curve_fit(pf_try21, xcurr1, ycurr1, p0 = [xdat[peak]-2, xdat[peak]+2, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr1[0], xcurr1[0], 0.005, 0.005], [xcurr1[-1], xcurr1[-1], 2, 2]))
            if ch == 2: args1 = scio.curve_fit(pf_try22, xcurr1, ycurr1, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr1[0], xcurr1[0], 0.005, 0.005], [xcurr1[-1], xcurr1[-1], 2, 2]))
            if ch == 3: args1 = scio.curve_fit(pf_try23, xcurr1, ycurr1, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr1[0], xcurr1[0], 0.005, 0.005], [xcurr1[-1], xcurr1[-1], 2, 2]))
            dt2cf1 += time() - st2cf1
        except:
            args1 = [[-1, -1, 5, 5], []]

        if ch == 0:   diff1 = sum([abs(val-bal)**2 for val,bal in zip(ycurr1, pf_try20(xcurr1,args1[0][0],args1[0][1],args1[0][2],args1[0][3]))])#**0.5
        elif ch == 1: diff1 = sum([abs(val-bal)**2 for val,bal in zip(ycurr1, pf_try21(xcurr1,args1[0][0],args1[0][1],args1[0][2],args1[0][3]))])#**0.5
        elif ch == 2: diff1 = sum([abs(val-bal)**2 for val,bal in zip(ycurr1, pf_try22(xcurr1,args1[0][0],args1[0][1],args1[0][2],args1[0][3]))])#**0.5
        elif ch == 3: diff1 = sum([abs(val-bal)**2 for val,bal in zip(ycurr1, pf_try23(xcurr1,args1[0][0],args1[0][1],args1[0][2],args1[0][3]))])#**0.5

        diff1 = diff1/((args1[0][2]**2+args1[0][3]**2)**0.5*len(xcurr1))

        if diff1 > 0.001:
            lst2 = 0
            rst2 = 0

            while peak-lst2>=1 and lst2<20 and ydat[peak-lst2-1]>=0.003 and not (lst2>5 and ydat[peak-lst2-2]>ydat[peak-lst2-1] and ydat[peak-lst2-3]>ydat[peak-lst2-2]):
                lst2 += 1

            while peak+rst2<=10000 and rst2<20 and ydat[peak+rst2+1]>=0.003 and not (rst2>5 and ydat[peak+rst2+2]>ydat[peak+rst2+1] and ydat[peak+rst2+3]>ydat[peak+rst2+2]):
                rst2 += 1

            lst2 += 1
            rst2 += 2
            
            if lst2+rst2 <= 5:
                args2 = [[-1, -1, 5, 5], []]
                diff2 = 10
            
            else:
                xcurr2 = xdat[peak-lst2:peak+rst2]
                ycurr2 = ydat[peak-lst2:peak+rst2]
                if len(xcurr2) == 0: xcurr2 = [-1]

                try:
                    st2cf2 = time()
                    if ch == 0: args2 = scio.curve_fit(pf_try20, xcurr2, ycurr2, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr2[0], xcurr2[0], 0, 0], [xcurr2[-1], xcurr2[-1], 2, 2]))
                    if ch == 1: args2 = scio.curve_fit(pf_try21, xcurr2, ycurr2, p0 = [xdat[peak]-2, xdat[peak]+2, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr2[0], xcurr2[0], 0, 0], [xcurr2[-1], xcurr2[-1], 2, 2]))
                    if ch == 2: args2 = scio.curve_fit(pf_try22, xcurr2, ycurr2, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr2[0], xcurr2[0], 0, 0], [xcurr2[-1], xcurr2[-1], 2, 2]))
                    if ch == 3: args2 = scio.curve_fit(pf_try23, xcurr2, ycurr2, p0 = [xdat[peak]-1, xdat[peak]+1, ydat[peak]/2 - 0.01, ydat[peak]/2 + 0.01], bounds = ([xcurr2[0], xcurr2[0], 0, 0], [xcurr2[-1], xcurr2[-1], 2, 2]))
                    dt2cf2 += time() - st2cf2
                except:
                    args2 = [[-1, -1, 5, 5], []]

                if ch == 0:   diff2 = sum([abs(val-bal)**2 for val,bal in zip(ycurr2, pf_try20(xcurr2,args2[0][0],args2[0][1],args2[0][2],args2[0][3]))])#**0.5
                elif ch == 1: diff2 = sum([abs(val-bal)**2 for val,bal in zip(ycurr2, pf_try21(xcurr2,args2[0][0],args2[0][1],args2[0][2],args2[0][3]))])#**0.5
                elif ch == 2: diff2 = sum([abs(val-bal)**2 for val,bal in zip(ycurr2, pf_try22(xcurr2,args2[0][0],args2[0][1],args2[0][2],args2[0][3]))])#**0.5
                elif ch == 3: diff2 = sum([abs(val-bal)**2 for val,bal in zip(ycurr2, pf_try23(xcurr2,args2[0][0],args2[0][1],args2[0][2],args2[0][3]))])#**0.5

                diff2 = diff2/((args2[0][2]**2+args2[0][3]**2)**0.5*len(xcurr2))

        else: diff2 = 10

        if diff1 <= diff2 or diff2 == 10: 
            args, diff = args1, diff1

        else:
            args, diff = args2, diff2
        
        dt2p += time() - st2p
        
        if args[0][0]<=args[0][1]:
            return [args[0][0], args[0][1], args[0][2], args[0][3]], diff#, dt2p, dt2cf1, dt2cf2
        
        else:
            return [args[0][1], args[0][0], args[0][3], args[0][2]], diff#, dt2p, dt2cf1, dt2cf2
    
    
    def ts_from_ms(self, ms, r = True):
        time = self.data[0][0]
        #tott = time[-1]-time[0]
        dt = time[1]-time[0]
        #steps = len(time)-1
        if r == False: return (ms-time[0])/dt
        else: return round((ms-time[0])/dt)
    
    
    def ms_from_ts(self, ts):
        time = self.data[0][0]
        dt = time[1]-time[0]
        return time[0]+ts*dt
    
    
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
        if int(self.df[self.df['LyA'] == self.filepath]['sc_pow']) == -3:
            return 1
        else:
            return 0
    
    
    def corr():
        '''

        the shape of the voltage pulse from 10 points before the peak to 1000 points after is taken. This is done for 4 files, each having only 1 pulse at around 0.1 V. We use the Lfile.parpeak function to get the 'actual'
        time and height of each pulse. The times are then shifted such that the actual pulse timing is at 0. The voltages are all normalized so that the actual pulse height is 0.1V. Then the datapoints from the 4 pulses are
        put together into an array. With this array we define a function, which first searches for the point in the combined array just before the input x. Then a polynomial of second degree is fit for the 9 points around
        the one we found (4 before to 4 after). The returned value is the value of the polynomial at x. We evaluate this function at 10000 points (from the the first timestep in the combined array until the last). This
        process is repeated for each channel.

        The arrays for each channel, together with the corresponding times, are all put into one array of shape (2, 4). In the first subarray are the times for each channel and in the second one are the voltages. This array
        is finally saved at '/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy'.

        These values can be used to eliminate the noise which comes with each pulse from the LyA trc files.

        The pulses are all for 2.2 kV MCP voltage, but the shape stays the same for other voltages.
        
        If the oscilloscope settings are changed, this correction probably is not valid anymore and a new one with the updated osci settings has to be created.
        
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
        ret = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy') #get the correction arrays

        return ret
    
    
    def get_events_from_file(self, hgt = 0.005, dist = 5, trange = None):
        '''
        get information about the pulses recorded by the 4 mcps in the lya setup for the file
        
        Parameters
        ------------
        hgt = 0.005 is the minimum height for a pulse
        dist = 5 the distance two pulses have to be apart from eachother
        trange = the time range from which we want pulses
            input as a list with trange[0] < trange[1], default is [0, 10002]
            default is None, in which case all pulses will be returned
        
        Returns
        ------------
        pt = list of times at which events were registered for each channel
        ph = list of heights of the events for each channel
        mw: 1 if the mw is on and 0 if it is off for the file
        self.run_number(): run in which the current file was taken, 0 if it was taken outside of any run
        
        '''
        dtf = 0
        dtfw = 0
        dtfc = 0
        dtfs = 0
        dtfp1 = 0
        dtfp2 = 0
        
        itp1 = 0
        itp2 = 0
        
        ap = 0
        bp = 0
        cp = 0
        
        stf = time()
        
        if self.valid == False:
            return [[],[],[],[]], [[],[],[],[]], self.mw_on(), self.run_number()

        pt = [[],[],[],[]] #list to put the lists where the events are into   
        ph = [[],[],[],[]] #list for the peak heights of the 4 detectors
        pp = [[],[],[],[]] #list for the plateau widths of the pulses
        mw = self.mw_on() #return mw = 1 if mw was on, otherwise mw = 0
        
        maxvol = 0.2906665578484535
            
        #corr = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy') #get the correction arrays
        #until how many timesteps after the peak we have to correct the voltage (changes depending on the channel)
        lims = [len(self.corr_df[self.corr_df['0'] != -100]), len(self.corr_df[self.corr_df['1'] != -100]), len(self.corr_df[self.corr_df['2'] != -100]), len(self.corr_df[self.corr_df['3'] != -100])] #how many elements after the peak we have to correct the voltage (changes depending on the channel)
        lims = [int((val-12001)/1000) for val in lims]
        
        o_data = self.read_trc() #original data from the trc file
        step_index = 1000
        
        #go through the 4 channels
        for i in [0, 1, 2, 3]:
            #go through pulses > 0.03V, after each pulse we subtract the correction from the current voltage channel, centered at the peak, and use sci.find_peaks for > 0.03V again until we find no more peaks
            #the timestep of the pulse max and the height from Lfile.parpeak are added into an array (pt and ph) to return them in the end
            peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = 0.03, distance = 1, wlen = 200, plateau_size = 1))
            
            ignore_pulses = []
            keep_vals = False
            
            xc_temp = self.corr_df[str(i)][:(lims[i]+12)*1000+1]
            yc_temp = self.corr_df[str(4+i)][:(lims[i]+12)*1000+1]
            
            while len(peaks[0]) > 0:
                stfw = time()
                curr_pos = 0
                currt = peaks[0][0]
                pp[i] += [peaks[1]['plateau_sizes'][0]]
                
                if keep_vals == False:
                    stfp1 = time()
                    t0, h0, d0 = self.parpeak(i, currt, edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                    dtfp1 += time() - stfp1
                    itp1 += 1
                    ap += 1
                    if d0 > 1e-4:
                        #print('a ', i, currt)
                        stfp2 = time()
                        args, diff2 = self.parpeak2(i, currt)
                        dtfp2 += time() - stfp2
                        itp2 += 1
                    else: args, diff2 = [-1, -1, 1, 1], 1
                                    
                if d0 > 1e-4 and diff2 < 0.001 and (diff2 < d0/2 if h0 >= maxvol else diff2 < d0/1.3) and abs(args[0]-args[1]) >= 1 and pp[i][-1] <= 3:
                    if args[2] >= args[3] and args[2] >= 0.01:
                        t0 = args[0]
                        h0 = args[2]
                    elif args[3] > args[2] and args[2] < 0.01:
                        t0 = args[1]
                        h0 = args[3]
                    else:
                        t0 = args[0]
                        h0 = args[2]
                
                currt = round(t0) 
                pt[i] += [currt]
                if h0 > maxvol: ph[i] += [maxvol]
                else: ph[i] += [h0]
                
                index0 = np.argmin([abs(val+t0-self.data[0][0][currt-10]) for val in xc_temp[:2000]])
                
                stfc = time()
                
                for z in range(min(lims[i]+10, len(self.data[0][0])-currt+10)):
                    if o_data[1][i][currt-10+z] == -maxvol and z > 16:
                        self.data[1][i][currt-10+z] = maxvol
                        
                    else:
                        if h0*yc_temp[index0+z*step_index] > maxvol: self.data[1][i][currt-10+z] -= maxvol
                        elif h0*yc_temp[index0+z*step_index] < -0.048: self.data[1][i][currt-10+z] -= -0.048
                        else: self.data[1][i][currt-10+z] -= h0*yc_temp[index0+z*step_index]
                            
                dtfc += time() - stfc
                stfs = time()
                
                keep_vals = False
                peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = 0.03, distance = 1, wlen = 200, plateau_size = 1))
                
                #if len(peaks[0]) > 0:
                #    
                #    stfp1 = time()
                #    t0, h0, d0 = self.parpeak(i, peaks[0][0], edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                #    dtfp1 += time() - stfp1
                #    itp1 += 1
                #    if d0 > 1e-4:
                #        #print('b ', i, peaks[0][0])
                #        stfp2 = time()
                #        args, diff2 = self.parpeak2(i, peaks[0][0])
                #        dtfp2 += time() - stfp2
                #        itp2 += 1
                #    else: args, diff2 = [-1, -1, 1, 1], 1
                                        
                while len(peaks[0]) > 0:
                    #print(ignore_pulses)
                    if peaks[0][0] in ignore_pulses:
                        peaks[0] = peaks[0][1:]
                        peaks[1]['peak_heights'] = peaks[1]['peak_heights'][1:]                                                                                            
                        peaks[1]['left_edges'] = peaks[1]['left_edges'][1:]
                        peaks[1]['right_edges'] = peaks[1]['right_edges'][1:]
                        peaks[1]['plateau_sizes'] = peaks[1]['plateau_sizes'][1:]
                    
                    else: 
                        stfp1 = time()
                        t0, h0, d0 = self.parpeak(i, peaks[0][0], edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                        dtfp1 += time() - stfp1
                        itp1 += 1
                        bp += 1
                        
                        if d0 > 1e-4:
                            #print('c ', i, peaks[0][0])
                            stfp2 = time()
                            args, diff2 = self.parpeak2(i, peaks[0][0])
                            dtfp2 += time() - stfp2
                            itp2 += 1
                        else: args, diff2 = [-1, -1, 1, 1], 1
                        
                        if d0 > 0.003 and (diff2 > 0.001 or abs(args[0]-args[1]) < 1):
                            ignore_pulses += [peaks[0][0]]
                            peaks[0] = peaks[0][1:]
                            peaks[1]['peak_heights'] = peaks[1]['peak_heights'][1:]                                                                                            
                            peaks[1]['left_edges'] = peaks[1]['left_edges'][1:]
                            peaks[1]['right_edges'] = peaks[1]['right_edges'][1:]
                            peaks[1]['plateau_sizes'] = peaks[1]['plateau_sizes'][1:]
                            
                        else:
                            keep_vals = True
                            break
                    
                dtfs += time() - stfs
                dtfw += time() - stfw
            
            #self.data[1][i] = [val if val > -0.048 else 0 for val in self.data[1][i]]
            
            #search for all pulses still in the voltage with height > hgt (many of these are still bad)
            peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = hgt, distance = dist, wlen = 200, plateau_size = 1))
            
            curr_pt = peaks[0]
            curr_ph = peaks[1]['peak_heights']
            curr_pp = peaks[1]['plateau_sizes']

            if len(pt[i]) == 0: pt[i] = [-1]
            
            for k in range(len(curr_pt)):
                #if the pulse height is below 0.03V and before the 3000th timestep we discard it
                if curr_pt[k] < 3000 and curr_ph[k] < 0.03:
                    curr_pt[k] = -1
                    curr_ph[k] = -1
                    curr_pp[k] = -1
                
                if curr_pt[k] != -1:
                    stfp1 = time()
                    t0, h0, d0 = self.parpeak(i, curr_pt[k])
                    dtfp1 += time() - stfp1
                    itp1 += 1
                    cp += 1
                else: t0, h0, d0 = -1, 1, 1
                
                currt = round(t0)
                
                if curr_pt[k] != -1 and 0.015 <= h0 < maxvol and d0/h0 > 0.015:
                    curr_pt[k] = -1
                    curr_ph[k] = -1
                    curr_pp[k] = -1
                    
                if curr_pt[k] != -1 and (h0 < 0.015 or min([abs(val-curr_pt[k]) for val in pt[i]]) <= 10):
                    index0 = np.argmin([abs(val+t0-self.data[0][0][currt-10]) for val in xc_temp[:2000]])
                    xcd = [val+t0 for val in xc_temp[2000+index0:29000+index0][::step_index]]
                    ycd = [val*h0 for val in yc_temp[2000+index0:29000+index0][::step_index]]
                    curr_diff = (sum([(val-bal)**2 for val,bal in zip(self.data[1][i][currt-8:currt+19],ycd)])/len(ycd))
                    if h0 < 0.008: curr_lim = 1.5e-4
                    elif h0 > 0.02: curr_lim = 5.5e-4
                    else: curr_lim = 2.5e-4
                    if curr_diff/h0 > curr_lim:
                        curr_pt[k] = -1
                        curr_ph[k] = -1
                        curr_pp[k] = -1
                        
                if curr_pt[k] != -1:
                    curr_pt[k] = currt
                    curr_ph[k] = h0 if h0 < maxvol else maxvol
            
            pt[i] += list(curr_pt)
            ph[i] += list(curr_ph)
            
            #only keep elements in pt, ph if they are not -1
            pt[i] = [val for val in pt[i] if val != -1]
            ph[i] = [val for val in ph[i] if val != -1]
            
            #sort pt[i] and ph[i] by pt[i]
            ph[i] = [bal for val,bal in sorted(zip(pt[i],ph[i]))]
            pt[i] = sorted(pt[i])
            
            #only keep peaks within trange
            if trange != None:
                ph[i] = [bal for val,bal in zip(pt[i],ph[i]) if trange[0] <= val <= trange[1]]
                pt[i] = [val for val in pt[i] if trange[0] <= val <= trange[1]]
                          
        dtf += time() - stf
        #print('HERE:', ap, bp, cp)
        return pt, ph, mw, self.run_number(), dtf, dtfc, dtfw, dtfs, dtfp1, dtfp2, itp1, itp2
    
    
    def dat(self):
        return self.data
    