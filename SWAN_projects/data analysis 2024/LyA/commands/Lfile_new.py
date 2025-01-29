#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/commands')
sys.path.append('/eos/user/l/lkoller/data-analysis-software')
sys.path.append('/eos/user/l/lkoller/SWAN_projects/commands/data_loader')

from pathlib import Path
import pandas as pd
import numpy as np
from readTrc_4CH import Trc
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


def get_corrfun_3dfit_values(ch):
    corr = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy')
    corr = corr = [corr[0][ch][40:160], -corr[1][ch][40:160]]
    time_steps = np.linspace(corr[0][0], corr[0][-1], 12)

    
    #if x == corr[0][-1]:
    #    return corr[1][-1]
    
    for i in range(len(time_steps)-1):
        curr_fit = np.polyfit([val for val in corr[0] if time_steps[i] <= val < time_steps[i+1]], [bal for val,bal in zip(corr[0],corr[1]) if time_steps[i] <= val < time_steps[i+1]], 3)
        #if time_steps[i] <= x < time_steps[i+1]:
    return   


def corr_func0(x, t, h):
    time_steps = np.linspace(-6.354787824327302e-09, 5.661687220735794e-09, 12)
    
    args = [[3.934639095070963e+23, 7034286886823345.0, 41854378.92450875, 0.08287320365215026],
[9.269994512155752e+24, 1.3922799082608405e+17, 697635618.8483183, 1.1666727760176734],
[-1.5026925880353366e+24, -2578996934104233.0, 67453748.15151764, 0.22349043306267863],
[-7.843778648545667e+24, -6.4148076453805096e+16, -135430589.1765476, -0.0033859005475927895],
[-7.464333648805273e+24, -3.757257219488522e+16, -39899355.64336706, 0.08488705412497824],
[-2.0211472694042374e+24, -8492998009241786.0, -1207899.9802789267, 0.10036344805341817],
[-1.5931433559681425e+24, 1895171031169829.8, -7984048.030594953, 0.10125047510595843],
[1.3600000507293238e+24, -6753875430619029.0, -822846.6630111501, 0.099958480439117],
[-2.4034331001504174e+24, 1.9619147006771244e+16, -69424210.93028319, 0.16453894923333626],
[-1.5600633937850955e+24, 1.5601032731662354e+16, -77447912.35981013, 0.20535870464296055],
[5.262767616478577e+24, -7.698032258090013e+16, 346836053.25736874, -0.4526503491327303]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret


def corr_func1(x, t, h):
    time_steps = np.linspace(-6.033656730400169e-09, 5.9765671846762e-09, 12)
    
    args = [[-6.108458475147061e+23, -9260385571152986.0, -45004625.47814578, -0.06863074454077511],
[-2.1643259018492525e+24, -2.144040428962637e+16, -45887418.78759391, 0.03759790893676375],
[-1.4792155075117346e+25, -1.4469731499137206e+17, -436624173.72320306, -0.36011592069845144],
[-1.2028271044998155e+25, -8.668233903779682e+16, -176520658.06412867, -0.02568191778818238],
[-2.3759145663101441e+24, -1.6027183346193324e+16, -13310660.282735229, 0.09501425531011455],
[1.650901881205521e+24, -3773699088532689.5, -898526.1708863726, 0.09898068850019656],
[-8.948902849971813e+23, 2245634050781790.0, -7682583.56046431, 0.10125434076808966],
[-2.0018861669019225e+24, 9747379133990118.0, -23784724.171171457, 0.11228004268267815],
[3.5922875297253994e+23, -1.2784272163272506e+16, 50672708.158802226, 0.028917842162788008],
[-5.384753211591939e+24, 6.4681733149562456e+16, -286001152.5517751, 0.5043697129863712],
[-1.7350312344137994e+24, 2.90319798401443e+16, -184108882.48177704, 0.4318990214587847]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret


def corr_func2(x, t, h):
    time_steps = np.linspace(-6.4766464713923734e-09, 5.537905149059421e-09, 12)
    
    args = [[-6.601276666250683e+23, -1.2230099401769524e+16, -74443793.96465358, -0.1484992084267505],
[3.7397372885235815e+24, 5.334511305842896e+16, 253334765.86425796, 0.4020359728025199],
[1.3704244620904398e+25, 1.669279685284471e+17, 684482704.3721521, 0.9482312884968197],
[9.014519693694474e+24, 8.249968318127117e+16, 285771248.101462, 0.3836093456997951],
[-7.5401375292223455e+22, -6781133261863830.0, 9225586.920326333, 0.11239299467558947],
[2.95184563618179e+24, -372978593507191.3, 195217.77116635017, 0.10005613011135402],
[1.1140462412672619e+24, -6643724666688025.0, -3920159.1465815715, 0.09978478256094188],
[3.4539993579465664e+24, -1.6780579666151678e+16, 10841498.138624784, 0.09285100121746509],
[4.994513564964777e+24, -4.019645040955628e+16, 91698113.05926123, 0.012199418347630808],
[4.404429900839818e+24, -5.227594003127282e+16, 185749960.13175076, -0.14528860071086697],
[1.370200803727519e+25, -1.9734619959470013e+17, 921070466.819911, -1.3644940235633685]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret


def corr_func3(x, t, h):
    time_steps = np.linspace(-6.274464642765424e-09, 5.740567833839197e-09, 12)
    
    args = [[-1.7670733730389408e+24, -3.1602381638490056e+16, -187446015.55394512, -0.3686603463919524],
[3.3134779759251175e+24, 5.237112629924486e+16, 279968129.2943164, 0.5058974773937585],
[2.551722165513196e+24, 3.1378770128331204e+16, 150823383.9231859, 0.2768132659141354],
[6.422710391005284e+24, 4.8731624421458424e+16, 151929128.00253493, 0.2290229469208394],
[3.286140880702547e+24, 7851473365599842.0, 24518494.6315547, 0.11280266583778845],
[3.5236573044970326e+24, -5326537317585951.0, -1268361.0083138596, 0.10013460148550317],
[-7.056978542581246e+23, -7743495765287535.0, -1024162.5476529318, 0.09959937512398769],
[4.183377441616648e+24, -2.683764902417701e+16, 32596736.007441316, 0.0771200125571626],
[9.698454304368224e+24, -8.920220933101048e+16, 249379137.51690757, -0.1610254726166941],
[9.623140042363136e+24, -1.1597513017341754e+17, 443506324.17545277, -0.5096534195128726],
[-6.00625838610502e+23, 1.268413606108325e+16, -97216190.063373, 0.25141528458960055]]
    
    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret


def corr_func(x, t, h, ch):
    if ch == 0:
        time_steps = np.linspace(-6.354787824327302e-09, 5.661687220735794e-09, 12)
        args = [[3.934639095070963e+23, 7034286886823345.0, 41854378.92450875, 0.08287320365215026],
[9.269994512155752e+24, 1.3922799082608405e+17, 697635618.8483183, 1.1666727760176734],
[-1.5026925880353366e+24, -2578996934104233.0, 67453748.15151764, 0.22349043306267863],
[-7.843778648545667e+24, -6.4148076453805096e+16, -135430589.1765476, -0.0033859005475927895],
[-7.464333648805273e+24, -3.757257219488522e+16, -39899355.64336706, 0.08488705412497824],
[-2.0211472694042374e+24, -8492998009241786.0, -1207899.9802789267, 0.10036344805341817],
[-1.5931433559681425e+24, 1895171031169829.8, -7984048.030594953, 0.10125047510595843],
[1.3600000507293238e+24, -6753875430619029.0, -822846.6630111501, 0.099958480439117],
[-2.4034331001504174e+24, 1.9619147006771244e+16, -69424210.93028319, 0.16453894923333626],
[-1.5600633937850955e+24, 1.5601032731662354e+16, -77447912.35981013, 0.20535870464296055],
[5.262767616478577e+24, -7.698032258090013e+16, 346836053.25736874, -0.4526503491327303]]
        
    if ch == 1:
        time_steps = np.linspace(-6.033656730400169e-09, 5.9765671846762e-09, 12)
        args = [[-6.108458475147061e+23, -9260385571152986.0, -45004625.47814578, -0.06863074454077511],
[-2.1643259018492525e+24, -2.144040428962637e+16, -45887418.78759391, 0.03759790893676375],
[-1.4792155075117346e+25, -1.4469731499137206e+17, -436624173.72320306, -0.36011592069845144],
[-1.2028271044998155e+25, -8.668233903779682e+16, -176520658.06412867, -0.02568191778818238],
[-2.3759145663101441e+24, -1.6027183346193324e+16, -13310660.282735229, 0.09501425531011455],
[1.650901881205521e+24, -3773699088532689.5, -898526.1708863726, 0.09898068850019656],
[-8.948902849971813e+23, 2245634050781790.0, -7682583.56046431, 0.10125434076808966],
[-2.0018861669019225e+24, 9747379133990118.0, -23784724.171171457, 0.11228004268267815],
[3.5922875297253994e+23, -1.2784272163272506e+16, 50672708.158802226, 0.028917842162788008],
[-5.384753211591939e+24, 6.4681733149562456e+16, -286001152.5517751, 0.5043697129863712],
[-1.7350312344137994e+24, 2.90319798401443e+16, -184108882.48177704, 0.4318990214587847]]
        
    if ch == 2:
        time_steps = np.linspace(-6.4766464713923734e-09, 5.537905149059421e-09, 12)
        args = [[-6.601276666250683e+23, -1.2230099401769524e+16, -74443793.96465358, -0.1484992084267505],
[3.7397372885235815e+24, 5.334511305842896e+16, 253334765.86425796, 0.4020359728025199],
[1.3704244620904398e+25, 1.669279685284471e+17, 684482704.3721521, 0.9482312884968197],
[9.014519693694474e+24, 8.249968318127117e+16, 285771248.101462, 0.3836093456997951],
[-7.5401375292223455e+22, -6781133261863830.0, 9225586.920326333, 0.11239299467558947],
[2.95184563618179e+24, -372978593507191.3, 195217.77116635017, 0.10005613011135402],
[1.1140462412672619e+24, -6643724666688025.0, -3920159.1465815715, 0.09978478256094188],
[3.4539993579465664e+24, -1.6780579666151678e+16, 10841498.138624784, 0.09285100121746509],
[4.994513564964777e+24, -4.019645040955628e+16, 91698113.05926123, 0.012199418347630808],
[4.404429900839818e+24, -5.227594003127282e+16, 185749960.13175076, -0.14528860071086697],
[1.370200803727519e+25, -1.9734619959470013e+17, 921070466.819911, -1.3644940235633685]]
    
    if ch == 3:
        time_steps = np.linspace(-6.274464642765424e-09, 5.740567833839197e-09, 12)
        args = [[-1.7670733730389408e+24, -3.1602381638490056e+16, -187446015.55394512, -0.3686603463919524],
[3.3134779759251175e+24, 5.237112629924486e+16, 279968129.2943164, 0.5058974773937585],
[2.551722165513196e+24, 3.1378770128331204e+16, 150823383.9231859, 0.2768132659141354],
[6.422710391005284e+24, 4.8731624421458424e+16, 151929128.00253493, 0.2290229469208394],
[3.286140880702547e+24, 7851473365599842.0, 24518494.6315547, 0.11280266583778845],
[3.5236573044970326e+24, -5326537317585951.0, -1268361.0083138596, 0.10013460148550317],
[-7.056978542581246e+23, -7743495765287535.0, -1024162.5476529318, 0.09959937512398769],
[4.183377441616648e+24, -2.683764902417701e+16, 32596736.007441316, 0.0771200125571626],
[9.698454304368224e+24, -8.920220933101048e+16, 249379137.51690757, -0.1610254726166941],
[9.623140042363136e+24, -1.1597513017341754e+17, 443506324.17545277, -0.5096534195128726],
[-6.00625838610502e+23, 1.268413606108325e+16, -97216190.063373, 0.25141528458960055]]

    ret = [0]*len(x)
    for k in range(len(x)):
        for i in range(len(time_steps)-1):
            if time_steps[i] <= x[k]-t < time_steps[i+1]:
                ret[k] = (args[i][0]*(x[k]-t)**3 + args[i][1]*(x[k]-t)**2 + args[i][2]*(x[k]-t) + args[i][3])*h/0.1
    
    return ret


def corr_func0_2peaks(x, t1, t2, h1, h2):
    if type(x) != type(np.array([])):
        return corr_func0([x], t1, h1) + corr_func0([x], t2, h2)

    res = [0]*len(x)
    res1 = corr_func0(x, t1, h1)
    res2 = corr_func0(x, t2, h2)
    for i in range(len(x)):
        res[i] = res1[i] + res2[i]
        
    return res


def corr_func1_2peaks(x, t1, t2, h1, h2):
    if type(x) != type(np.array([])):
        return corr_func1([x], t1, h1) + corr_func1([x], t2, h2)
    res = [0]*len(x)
    res1 = corr_func1(x, t1, h1)
    res2 = corr_func1(x, t2, h2)
    for i in range(len(x)):
        res[i] = res1[i] + res2[i]
        
    return res


def corr_func2_2peaks(x, t1, t2, h1, h2):
    if type(x) != type(np.array([])):
        return corr_func2([x], t1, h1) + corr_func2([x], t2, h2)
    
    res = [0]*len(x)
    res1 = corr_func2(x, t1, h1)
    res2 = corr_func2(x, t2, h2)
    for i in range(len(x)):
        res[i] = res1[i] + res2[i]
        
    return res


def corr_fun3_2peaks(x, t1, t2, h1, h2):
    if type(x) != type(np.array([])):
        return corr_func0([x], t1, h1) + corr_func3([x], t2, h2)
    
    res = [0]*len(x)
    res1 = corr_func3(x, t1, h1)
    res2 = corr_func3(x, t2, h2)
    for i in range(len(x)):
        res[i] = res1[i] + res2[i]
        
    return res


def find_values_2peaks(peak, xdat, ydat, ch):
    xcurr = xdat[peak-8:peak+8]
    ycurr = ydat[peak-8:peak+8]
    
    if ch == 0: args = scio.curve_fit(corr_func0_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
    if ch == 1: args = scio.curve_fit(corr_func1_2peaks, xcurr, ycurr, p0 = [xdat[peak-3], xdat[peak+3], ydat[peak-3], ydat[peak+3]], bounds = ([xcurr[0], xcurr[0], -1, -1], [xcurr[-1], xcurr[-1], -0.001, -0.001]))
    if ch == 2: args = scio.curve_fit(corr_func2_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
    if ch == 3: args = scio.curve_fit(corr_func3_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])

    
    return args[0][0], args[0][1], args[0][2], args[0][3]





def find_values0_2peaks(peak, xdat, ydat):
    xcurr = xdat[peak-10:peak+11]
    ycurr = ydat[peak-10:peak+11]
    
    args = scio.curve_fit(corr_func0_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
    
    return args


def find_values1_2peaks(peak, xdat, ydat):
    xcurr = xdat[peak-10:peak+11]
    ycurr = ydat[peak-10:peak+11]
    
    args = scio.curve_fit(corr_func1_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
    
    return args


def corr_func0_3peaks(x, t1, t2, t3, h1, h2, h3):
    if type(x) != type(np.array([])):
        return corr_func0([x], t1, h1) + corr_func0([x], t2, h2) + corr_func0([x], t3, h3)
    
    res = [0]*len(x)
    res1 = corr_func0(x, t1, h1)
    res2 = corr_func0(x, t2, h2)
    res3 = corr_func0(x, t3, h3)
    for i in range(len(x)):
        res[i] = res1[i] + res2[i] + res3[i]
        
    return res

def find_values0_3peaks(peak, xdat, ydat):
    xcurr = xdat[peak-10:peak+11]
    ycurr = ydat[peak-10:peak+11]
    
    args = scio.curve_fit(corr_func0_3peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], xdat[peak], ydat[peak], ydat[peak], ydat[peak]])
    
    return args
    



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
        
        try:
            self.data = list(Trc().open(Path(self.filepath)))
        except:
            self.valid = False
            print('Trc readout error for file ', self.filepath)
        else:
            self.valid = True
            self.data[1] = [-self.data[1][0], -self.data[1][1], -self.data[1][2], -self.data[1][3]] #change the sign of the voltages (data[1][0-3]) so we can search for maxima peaks  
            for i in range(4):
                av = np.average(self.data[1][i][8000:10000])
                self.data[1][i] = [val - av for val in self.data[1][i]]
        if type(df) != type(None):
            self.df = df
        else:
            self.df = pd.read_csv('/eos/user/l/lkoller/GBAR/data24/datafile24.txt', delimiter = '\t')
        
    def __str__(self):
        return f'{self.filepath}'
    
    
    def parpeak(self, ch, peak, edges = [0, 0]):
        '''

        get the actual peak and timing for a voltage pulse from the LyA trc files by fitting a parabola to the 3 elements before the highest measured voltage and 1 point after
        if the parabola has a loweer peak than the voltage point, we keep the voltage as the pulse height

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
        xdat = self.data[0][0]
        ydat = self.data[1][ch]
        
        if edges[1]-edges[0] <= 2:
            xcurr = xdat[peak-5:peak+6]
            ycurr = ydat[peak-5:peak+6]

        else:
            xcurr = np.array(list(xdat[edges[0]-3:edges[0]]) + list(xdat[edges[1]:edges[1]+4]))
            ycurr = np.array(list(ydat[edges[0]-3:edges[0]]) + list(ydat[edges[1]:edges[1]+4]))

        try:
            if ch == 0: args = scio.curve_fit(corr_func0, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]])
            elif ch == 1: args = scio.curve_fit(corr_func1, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]])
            elif ch == 2: args = scio.curve_fit(corr_func2, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]])
            elif ch == 3: args = scio.curve_fit(corr_func3, xcurr, ycurr, p0 = [xdat[peak], ydat[peak]])
        except: 
            t, h = xdat[peak], ydat[peak]
        else: t, h = args[0][0], args[0][1]

        return t, h
    
    
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
        xdat = self.data[0][0]
        ydat = self.data[1][ch]
        
        xcurr = xdat[peak-10:peak+11]
        ycurr = ydat[peak-10:peak+11]

        try:
            if ch == 0: args = scio.curve_fit(corr_func0_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
            if ch == 1: args = scio.curve_fit(corr_func1_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
            if ch == 2: args = scio.curve_fit(corr_func2_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
            if ch == 3: args = scio.curve_fit(corr_func3_2peaks, xcurr, ycurr, p0 = [xdat[peak], xdat[peak], ydat[peak], ydat[peak]])
        except:
            return xcurr[peak], xcurr[peak], 1, 1

        return args[0][0], args[0][1], args[0][2], args[0][3]
    
    
    def ts_from_ms(self, ms):
        time = self.data[0][0]
        dt = time[1]-time[0]
        return round((ms-time[0])/dt)
    
    
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
        data = self.data
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
    
    
    def get_events_from_file(self, hgt = 0.005, dist = 5, trange = None, ch = 0, pl = False, ppt = -2, stop = 0, plt_xlims = [4700, 7000]):
        '''
        get information about the events recorded by the 4 mcps in the lya setup for the file
        
        Parameters
        ------------
        hgt = 0.005 is the minimum height for a peak to be recorded
        dist = 5 the distance two peaks have to be apart from eachother
        trange = the time range from which we want peaks
            input as a list with trange[0] < trange[1], default is [0, 10002] 
        
        Returns
        ------------
        events = list of times at which events were registered for each channel
        p_height = list of heights of the events for each channel
        self.mw_on() is 1 if the mw is on and 0 if it is off for the file
        
        '''        
        if self.valid == False:
            return [[],[],[],[]], [[],[],[],[]], self.mw_on(), self.run_number()
        
        
        pt = [[],[],[],[]] #list to put the lists where the events are into   
        ph = [[],[],[],[]] #list for the peak heights of the 4 detectors
        pp = [[],[],[],[]] #list for the plateau widths of the pulses
        mw = self.mw_on() #return mw = 1 if mw was on, otherwise mw = 0
        
        maxvol = 0.2906665578484535
            
        corr = np.load('/eos/user/l/lkoller/SWAN_projects/data analysis 2024/LyA/special stuff/2.2kV_volt_noise.npy') #get the correction arrays
        lims = [800, 600, 500, 600] #how many elements after the peak we have to correct the voltage (changes depending on the channel)
        
        o_data = self.read_trc() #original data from the trc file
        
        #go through the 4 channels
        for i in [0,1,2,3]:
            #go through pulses > 0.03V, after each pulse we subtract the correction from the current voltage channel, centered at the peak, and use sci.find_peaks for > 0.03V again until we find no more peaks
            #the timestep of the pulse max and the height from Lfile.parpeak are added into an array (pt and ph) to return them in the end
            peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = 0.03, distance = 1, wlen = 200, plateau_size = 1))
            #return peaks
            
            err = 0
            while len(peaks[0]) > 0 and err < 1:
                curr_pos = 0
                currt = peaks[0][0]
                t0, h0 = self.parpeak(i, currt, edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                pt[i] += [currt]
                if h0 > maxvol: ph[i] += [maxvol]
                else: ph[i] += [h0]
                pp[i] += [peaks[1]['plateau_sizes'][0]]
            

                
                for z in range(min(lims[i], len(self.data[0][0])-currt+10)):
                    for k in range(len(corr[0][i][curr_pos:])):
                        if curr_pos+k == len(corr[0][i])-1 or abs(corr[0][i][curr_pos+k]-self.data[0][0][z+currt-10]+t0) < abs(corr[0][i][curr_pos+k+1]-self.data[0][0][z+currt-10]+t0):
                            curr_pos += k
                            index = curr_pos
                            break
                    if z > -1:
                        if (o_data[1][i][z+currt-10] == -maxvol or self.data[1][i][z+currt-10]+corr[1][i][index]*h0/0.1 > maxvol) and corr[0][i][index] > 1e-8:
                            self.data[1][i][z+currt-10] = -o_data[1][i][z+currt-10]
                        else:
                            if corr[1][i][index]*h0/0.1 > 0.048: self.data[1][i][z+currt-10] += 0.048
                            elif corr[1][i][index]*h0/0.1 < -maxvol: self.data[1][i][z+currt-10] += -maxvol
                            else: self.data[1][i][z+currt-10] += corr[1][i][index]*h0/0.1
                                    
                err += stop

                
                t_control = peaks[0][0]
                peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = 0.03, distance = 1, wlen = 200, plateau_size = 1))
                
                if len(peaks[0]) > 0:
                    t0, h0 = self.parpeak(i, peaks[0][0], edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                    if peaks[1]['plateau_sizes'][0] <= 2:
                        diff = corr_func([self.data[0][0][peaks[0][0]-3], self.data[0][0][peaks[0][0]-2], self.data[0][0][peaks[0][0]-1], self.data[0][0][peaks[0][0]], self.data[0][0][peaks[0][0]+1], self.data[0][0][peaks[0][0]+2]], t0, h0, i)
                        diff = sum([abs(val-bal)**2 for val,bal in zip(diff,self.data[1][i][peaks[0][0]-3:peaks[0][0]+3])])
                    else:
                        diff = corr_func([self.data[0][0][peaks[1]['left_edges'][0]-2], self.data[0][0][peaks[1]['left_edges'][0]-1], self.data[0][0][peaks[1]['left_edges'][0]], self.data[0][0][peaks[1]['right_edges'][0]], self.data[0][0][peaks[1]['right_edges'][0]+1], self.data[0][0][peaks[1]['right_edges'][0]+2]], t0, h0, i)
                        diff = sum([abs(val-bal)**2 for val,bal in zip(diff,self.data[1][i][peaks[1]['left_edges'][0]-2:peaks[1]['left_edges'][0]+1]+self.data[1][i][peaks[1]['right_edges'][0]:peaks[1]['right_edges'][0]+3])])
                
                #it often happens that just before or after a peak, the correction does not line up properly and it looks like another peak so we ignore it
                #while len(peaks[0]) > 0 and (peaks[0][0] <= t_control or (peaks[0][0]-t_control < 20 and peaks[1]['peak_heights'][0] < ph[i][-1]/3)):
                while len(peaks[0]) > 0 and diff/h0**2 >= 0.2:

                    peaks[0] = peaks[0][1:]
                    peaks[1]['peak_heights'] = peaks[1]['peak_heights'][1:]                                                                                            
                    peaks[1]['left_edges'] = peaks[1]['left_edges'][1:]
                    peaks[1]['right_edges'] = peaks[1]['right_edges'][1:]
                    peaks[1]['plateau_sizes'] = peaks[1]['plateau_sizes'][1:]
                    
                    if len(peaks[0]) == 0: break
                    
                    t0, h0 = self.parpeak(i, peaks[0][0], edges = [peaks[1]['left_edges'][0], peaks[1]['right_edges'][0]])
                    if peaks[1]['plateau_sizes'][0] <= 2:
                        diff = corr_func([self.data[0][0][peaks[0][0]-3], self.data[0][0][peaks[0][0]-2], self.data[0][0][peaks[0][0]-1], self.data[0][0][peaks[0][0]], self.data[0][0][peaks[0][0]+1], self.data[0][0][peaks[0][0]+2]], t0, h0, i)
                        diff = sum([abs(val-bal)**2 for val,bal in zip(diff,self.data[1][i][peaks[0][0]-3:peaks[0][0]+3])])
                    else:
                        diff = corr_func([self.data[0][0][peaks[1]['left_edges'][0]-2], self.data[0][0][peaks[1]['left_edges'][0]-1], self.data[0][0][peaks[1]['left_edges'][0]], self.data[0][0][peaks[1]['right_edges'][0]], self.data[0][0][peaks[1]['right_edges'][0]+1], self.data[0][0][peaks[1]['right_edges'][0]+2]], t0, h0, i)
                        diff = sum([abs(val-bal)**2 for val,bal in zip(diff,self.data[1][i][peaks[1]['left_edges'][0]-2:peaks[1]['left_edges'][0]+1]+self.data[1][i][peaks[1]['right_edges'][0]:peaks[1]['right_edges'][0]+3])])

            #search for all pulses still in the voltage with height > hgt (many of these are still bad)
            peaks = list(sci.find_peaks(self.data[1][i], prominence = 0.001, height = hgt, distance = dist, wlen = 200, plateau_size = 1))
            
            #if there is a pulse closer than 10 elements to one already in pt[i], we discard it
            if len(pt[i]) > 0:
                for k in range(len(peaks[0])):
                    if min([abs(val-peaks[0][k]) for val in pt[i]]) <= 10:
                        peaks[0][k] = -1
                        peaks[1]['peak_heights'][k] = -1
                        peaks[1]['plateau_sizes'][k] = -1
            
            pto = pt[i].copy()
            pho = ph[i].copy()
            pt[i] += list(peaks[0])
            ph[i] += list(peaks[1]['peak_heights'])
            pp[i] += list(peaks[1]['plateau_sizes'])
            
            #sort pt and ph by time
            pp[i] = [bal for val,bal in sorted(zip(pt[i],pp[i]))]
            ph[i] = [bal for val,bal in sorted(zip(pt[i],ph[i]))]
            pt[i] = sorted(pt[i])


            for k in range(len(pt[i])):
                #if the pulse height is below 0.03V and before the 3000th timestep we discard it
                if pt[i][k] < 3000 and ph[i][k] < 0.03:
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
                    
                #if the standard deviation of the 30 to 10 elements before + 10 to 20 after the pulse > pulse height*0.3 and it is not in pto, we discard it
                tl = self.data[1][i][(pt[i][k]-30) : (pt[i][k]-10)] + self.data[1][i][(pt[i][k]+10) : (pt[i][k]+20)] #list with voltage for the datapoints from l before the mth peak of the ith channel until 10 before the peak
                if len(tl) == 0: tl = [0,1] #if the length of tl is 0, we define it as [0,1]
                mean = sum(tl) / len(tl) #mean of the elements in tl
                variance = sum([((x - mean) ** 2) for x in tl]) / len(tl) #variance of the elements in tl
                std = variance ** 0.5 #standard deviation of the elements in tl
                if not pt[i][k] in pto and std > ph[i][k]*0.3:
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
                    

                    
                #if the peak is below 0.02V and less than 40 timesteps after a pulse with height > 0.08V, we discard it
                if ph[i][k] < 0.02 and len([1 for val,bal in zip(pt[i],ph[i]) if (0 < pt[i][k]-val < 40 and bal > 0.08)]):
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
                    
                #if there is a pulse with height maxvol with a plateau width > 2, we discard pulses in the next 300 timesteps with height < 0.015V
                if ph[i][k] < 0.015 and len([1 for val,bal,kal in zip(pt[i],ph[i],pp[i]) if (0 < pt[i][k]-val < 300 and bal == maxvol and kal > 2)]) > 0:
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
                    
                #if there is a pulse with height > 0.04, we discard pulses in the next 300 timesteps with height < 0.01V
                if ph[i][k] < 0.01 and len([1 for val,bal,kal in zip(pt[i],ph[i],pp[i]) if (0 < pt[i][k]-val < 300 and bal > 0.04)]) > 0:
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
                    
                #if there is a pulse with height < 0.01 we discard pulses which have voltage elements with abs value > half the pulse height in self.data[1][i][pt[i][k]-10:pt[i][k]-3]+self.data[1][i][pt[i][k]+4:pt[i][k]+11]
                if ph[i][k] < 0.01 and max([abs(val) for val in self.data[1][i][pt[i][k]-12:pt[i][k]-5]+self.data[1][i][pt[i][k]+6:pt[i][k]+13]]) > 0.5*ph[i][k]:
                    pt[i][k] = -1
                    ph[i][k] = -1
                    pp[i][k] = -1
            
            #only keep elements in pt, ph if they are not -1
            pt[i] = [val for val in pt[i] if val != -1]
            ph[i] = [val for val in ph[i] if val != -1]
            
            if trange != None:
                ph[i] = [bal for val,bal in zip(pt[i],ph[i]) if trange[0] <= val <= trange[1]]
                pt[i] = [val for val in pt[i] if trange[0] <= val <= trange[1]]
            
            

            
        d = str(datetime.fromtimestamp(int(self.filepath[-18:-8])).date())
        if d in ['2024-04-23', '2024-04-25']:
            return pt, ph, mw, [0] * len(events)

        return pt, ph, mw, self.run_number()
    
    
    def dat(self):
        return self.data
    
#bad files (ch1): /eos/experiment/gbar/pgunpc/data/24_06_19/24_06_19lya/LY1234.1718762456.111.trc
#bad files (ch2): /eos/experiment/gbar/pgunpc/data/24_06_19/24_06_19lya/LY1234.1718765244.614.trc