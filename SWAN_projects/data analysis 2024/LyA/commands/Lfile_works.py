#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


def corr_func0(x, t, h):
    t -= 4999.681218121812
    time_steps = np.linspace(4993.399353426943, 5005.415828811855, 12)
    
    args = [[0.00039312109332297664, -5.889498240708686, 29410.9457534973, -48957409.321626626],
[0.009274674601533816, -138.97398169769974, 694139.9900063889, -1155684731.4338102],
[-0.0015045112537842677, 22.563960290485067, -112801.19303973536, 187970864.89937487],
[-0.007858789821415281, 117.81179646678622, -588708.8967618232, 980598299.7754327],
[-0.007457251012360087, 111.8157206554579, -558863.4403039913, 931080561.7512416],
[-0.002018429547403887, 30.26646277150783, -151282.41740946576, 252054211.28464162],
[-0.001593538180616412, 23.903793345722583, -119522.57725227084, 199210325.2947193],
[0.001361376674491478, -20.426407254940855, 102160.81807486198, -170315993.21285954],
[-0.002410130953206124, 36.16986550844983, -180938.89395532312, 301714201.3648892],
[-0.0015605689897930572, 23.42299043009167, -117187.30014333245, 195432863.87575835],
[0.0052743999246206074, -79.18926993273949, 396313.01576896355, -661133321.4772547]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret

def corr_func1(x, t, h):
    t -= 4999.642364236423
    time_steps = np.linspace(4993.720484529952, 5005.7307087847, 12)
    
    args = [[-0.0006073563034880027, 9.100693970406152, -45455.26626405301, 75678519.91651216],
[-0.002168664486628264, 32.50686999946263, -162418.92060080892, 270505913.8705665],
[-0.014743702776115834, 221.0004426093778, -1104227.2265690719, 1839087914.1510143],
[-0.01204969091473433, 180.64965663322707, -902769.9694131858, 1503819795.5022612],
[-0.0023740074369239255, 35.592340589772995, -177872.86973858325, 296306763.6548243],
[0.0016526995635683871, -24.793048120616938, 123978.01148386142, -206651299.75137588],
[-0.0008902566190207109, 13.355423491499552, -66784.9952205167, 111321466.29216954],
[-0.002000974891230075, 30.022888127412337, -150155.7837734572, 250328577.19271904],
[0.00035925814217764823, -5.401391732467484, 27069.601113859495, -45220479.989208594],
[-0.005375582823408575, 80.69433743659286, -403774.91730392183, 673464003.9680805],
[-0.0017336685390652346, 26.032758948607256, -130302.61909235586, 217402689.51792142]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret

def corr_func2(x, t, h):
    t -= 4999.830683068307
    time_steps = np.linspace(4993.277494776431, 5005.292046736678, 12)
    
    args = [[-0.0006611079476927215, 9.903883665106749, -49455.82126818698, 82320508.00706089],
[0.0037441692851449223, -56.10636830079403, 280251.2672079963, -466618288.6950687],
[0.013672058572041861, -204.90421919463105, 1023638.5666298164, -1704594673.6636105],
[0.009010124929678813, -135.06276081096667, 674868.566261553, -1124039426.7845442],
[-6.681340363108274e-05, 0.9954100317939992, -4943.089088502096, 8181870.215841246],
[0.002975902671084972, -44.63668283162968, 223174.12856704395, -371941405.8299663],
[0.0011152654324365692, -16.734803856453766, 83703.12414605, -139553703.27504057],
[0.003459144046810254, -51.901418269140336, 259578.3824354579, -432749461.20552844],
[0.0050002086045183745, -75.03968593306375, 375381.28694388084, -625940361.925315],
[0.0044059700501352566, -66.13859618087046, 330938.36896272306, -551973196.6615677],
[0.01368855329331251, -205.5153420909065, 1028512.7494643942, -1715749357.8615704]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret

def corr_func3(x, t, h):
    t -= 4999.642364236423
    time_steps = np.linspace(4993.479676610776, 5005.494709427189, 12)
    
    args = [[-0.001767534587164843, 26.480104227853044, -132236.15160336305, 220119975.29929358],
[0.0033176211525655325, -49.70944286438827, 248273.14880497946, -413332315.90748996],
[0.0025513714531229135, -38.23731653527269, 191020.47305693317, -318090883.22742563],
[0.006442100345890062, -96.5778841612136, 482621.4931216469, -803922904.5443966],
[0.00328186156386882, -49.21766773775174, 246037.08903318198, -409976447.08637905],
[0.003517615560092423, -52.76696749230616, 263848.5046658256, -439770280.93347335],
[-0.0007064598477755343, 10.58863485629302, -52901.86493988245, 88100934.36289766],
[0.004194571664181113, -62.942384013461464, 314830.9855748518, -524916785.476839],
[0.009687182403093544, -145.3896908455957, 727358.4352393423, -1212947705.5482337],
[0.009621818942593124, -144.4361431434793, 722725.3988827732, -1205450784.058386],
[-0.0006000277498621929, 9.012648487662979, -45124.4946811294, 75309730.17504613]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
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


# In[5]:


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
            
        corr = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy') #get the correction arrays
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
                    if args[2] >= args[3] and args[2] >= 0.02:
                        t0 = args[0]
                        h0 = args[2]
                    elif args[3] > args[2] and args[3] >= 0.02:
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
                        elif h0*yc_temp[index0+z*step_index] < -0.048: self.data[1][i][currt-10+z] -= 0
                        else: self.data[1][i][currt-10+z] -= h0*yc_temp[index0+z*step_index]
                            
                dtfc += time() - stfc
                stfs = time()
                
                keep_vals = False
                peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = 0.03, distance = 1, wlen = 200, plateau_size = 1))
                                        
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
            
            self.data[1][i] = [val if val > -0.048 else 0 for val in self.data[1][i]]
            
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
    

