#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects, center_of_mass
from PIL import Image
from pathlib import Path


# In[107]:


def cfinder(filepath, min_size = 5, min_sum = 0, thresh = None):
    '''

    get all the clusters in a tif file

    Parameters
    ------------
    filepath = path on the eos to the picture file
    min_size = 5 the minimum size a cluster has to be
    min_sum = 0 the minimum sum of the elements in a cluster
    thresh = None the value a pixel need for it to be considered part of a cluster
        if thresh == None, thresh is set to the average + 3* the standard deviation of the matrix of the picture
        
    Pedestal file for CMOS tracker: /eos/experiment/gbar/pgunpc/data/24_06_20/BAU-TRK_exp_100_us_G16_1718873546.147.ped.tif

    Returns
    ------------
    df = dataframe with the columns:
        cluster = matrix which contains the cluster shape
        pos = position of upper left element of the cluster shape in the matrix of the file
        size = number of elements in the cluster
        sum = sum of elements in the cluster

    '''
    mat = np.array(Image.open(Path(filepath)))
    thresh = thresh if thresh != None else (np.average(mat) + 3 * np.std(mat))
    mat[mat<=thresh] = 0
    
    #ped_clus_pos = [[113, 2401], [125, 1058], [232, 716], [263, 2205], [264, 1435], [318, 1821], [453, 1631], [512, 209], [539, 1986], [545, 2429], [587, 1293], [626, 743], [804, 489], [945, 2079], [953, 2437], [1153, 2388], [1156, 122], [1160, 1621], [1170, 2026], [1192, 1323], [1214, 1985], [1251, 697], [1252, 1249], [1281, 1223], [1312, 1871], [1323, 1374], [1327, 107], [1546, 1426], [1553, 1717], [1574, 72], [1608, 2204], [1624, 1409], [1628, 1170], [1673, 1099], [1674, 2270], [1687, 658], [1763, 1307], [1764, 1632], [1880, 147], [1918, 1977], [2036, 1998]]
    #for i in ped_clus_pos:
    #    mat[i[0],i[1]] = 0

    clus = []
    pos = []
    size = []
    tot = []

    struc = np.array([[1,1,1],[1,1,1],[1,1,1]]) * 0.1
    mat_labeled, num = label(mat, structure = struc)
    clust_slices = find_objects(mat_labeled)

    for i in range(num):
        temp_clus = mat[clust_slices[i]]
        temp_lab = mat_labeled[clust_slices[i]].tolist()
        temp_lab = [[val if val == i + 1 else 0 for val in bal] for bal in temp_lab]
        temp_size = len([val for bal in temp_lab for val in bal if val == i + 1])
        temp_tot = sum(sum(temp_clus))
        if temp_size >= min_size and temp_tot >= min_sum:
            size += [int(temp_size)]
            clus += [temp_clus.tolist()]
            pos += [[clust_slices[i][0].start, clust_slices[i][1].start]]
            tot += [temp_tot]

    df = pd.DataFrame([clus,pos,size,tot], index = ['cluster', 'pos', 'size', 'sum']).transpose()

    return df
