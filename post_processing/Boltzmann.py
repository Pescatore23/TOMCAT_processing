# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:13:39 2019

TOMCAT Boltzmann transforms

@author: firo
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import statsmodels.api as sma
lowess = sma.nonparametric.lowess
matplotlib.rcParams.update({'font.size': 26})

px=2.75E-6

Data=pickle.load(open(r"H:\11_Essential_Data\03_TOMCAT\02_Saturation_Curves\dynamic_data_full_v2_2.p","rb"))
 
sample=list(Data.keys())[3]
water_volume=Data[sample]['water_volume']


slice_waterpixel=Data[sample]['slice_waterpixel']

t=np.arange(slice_waterpixel.shape[1])

waterpixel=slice_waterpixel[:1298,:]
x=np.array(waterpixel.shape[0]-np.arange(waterpixel.shape[0]))

n=0.3
n=0
t0=25
z_r=0
z_r_old=0
t_max=85
tt=0
x0=0

X0=[0]
N=[0.1,0.2,0.3,0.4,0.5,0.6,0.8,1]
T0=[25]
N=[1]

for n in N:
    for t0 in T0:
        for x0 in X0:
            plt.figure()
            for ts in range(15,t_max):
#                if ts <t0: continue
#                prof_filt=lowess(waterpixel[:,ts],x,frac=0.1)
            #    if ts>24:
            #        n=0.66
            #        tt=-25+411-136
            #        z_r=-(len(x)-prof_filt[:,0]-250+xx)/(time_calib[23]-t_0+tt)**n1
                    
            #    if ts>33:
            #        n=0.2
            #        tt=-25+616-411
            #        z_r=(len(x)-prof_filt[:,0]-250+xx)/(time_calib[23]-t_0+tt)**n1+(len(x)-prof_filt[:,0]-250+xx)/(time_calib[32]-t_0+tt)**0.66
            
    #            chi=(prof_filt[:,0]-x0)/(ts-t0+tt)**n-z_r
    #            plt.plot(prof_filt[:,1],chi,color=matplotlib.cm.rainbow((ts-t0)/(t_max-t0)))
                
                n=0.33
                n=0.175
                t0=68
                
#                z_r=-(x-x0)/(69-48)**0.28-(x-x0)/(48-8)**2.73
                
                
                if ts<70:
                    n=0.28
                    t0=48
#                    z_r=-(x-x0)/(48-8)**2.73
                if ts<48:
                    t0=10
                    n=3.3
                    n=1
                    z_r=0
    
                chi=(x-x0)/(ts-t0)**n+z_r
                plt.plot(waterpixel[:,ts]*px**3*997,chi,color=matplotlib.cm.rainbow((ts-20)/(t_max-20)))
                plt.title('n1 = 3.3 t1 = 0, n2 = 0.28 t2 = 48, n3 = 0.175 t3 = 68')
                plt.xlabel('water mass [kg]')
                plt.ylabel('Xi')
                plt.ylim(0,70)