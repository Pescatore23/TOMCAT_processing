# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:44:43 2019

Correlation of time steps to actual time in TOMCAT 3 because of variable acquisition sequences


@author: firo
"""
import numpy as np


seq1=[]                                             #1 s/scan
seq1.append(1)                                      #0 180 1
seq1.append(10+seq1[-1])                            #3420 180 1
seq1=seq1+list(np.arange(1,61)*5+seq1[-1])          #1620 180 60
seq1=seq1+list(np.arange(1,121)*1+seq1[-1])         #180 180 120
seq1=seq1+list(np.arange(1,79)*10+seq1[-1])         #3420 180 78

f2=0.4
seq2=[] 	                                          #0.4 s/scan
seq2.append(f2)                                     #0 180 1
seq2=seq2+list(np.arange(1,151)*1*f2+seq2[-1])      #180 180 150
seq2=seq2+list(np.arange(1,31)*10*f2+seq2[-1])      #3420 180 30


seq3=[]                                             #1 s/scan
seq3.append(1)                                      #0 180 1
seq3=seq3+list(np.arange(1,61)*7+seq3[-1])          #2340 180 60
seq3=seq3+list(np.arange(1,121)*1+seq3[-1])         #180 180 120
seq3=seq3+list(np.arange(1,41)*7+seq3[-1])          #2340 180 40


seq4=[]                                             #1 s/scan
seq4.append(1)                                      #0 180 1
seq4=seq4+list(np.arange(1,81)*5+seq4[-1])          #1620 180 80
seq4=seq4+list(np.arange(1,181)*1+seq4[-1])         #180 180 180
seq4=seq4+list(np.arange(1,61)*7+seq4[-1])          #2340 180 60


seq5=[]                                             #1 s/scan
seq5.append(1)                                      #0 180 1
seq5=seq5+list(np.arange(1,81)*5+seq5[-1])          #1620 180 80
seq5=seq5+list(np.arange(1,181)*1+seq5[-1])         #180 180 180
seq5=seq5+list(np.arange(1,31)*7+seq5[-1])          #2340 180 30


seq6=[]                                             #0.4 s/scan
seq6.append(f2)                                     #0 180 1
seq6=seq6+list(np.arange(1,151)*1*f2+seq6[-1])      #180 180 150
seq6=seq6+list(np.arange(1,31)*1*f2+seq6[-1])       #180 180 30


seq7=[]                                             #0.4 s/scan
seq7.append(f2)                                     #0 180 1
seq7=seq7+list(np.arange(1,151)*1*f2+seq7[-1])      #180 180 150
seq7=seq7+list(np.arange(1,16)*10*f2+seq7[-1])      #3420 180 15


seq8=[]                                             #1 s/scan
seq8.append(1)                                      #0 180 1
seq8=seq8+list(np.arange(1,81)*5+seq8[-1])          #1620 180 80
seq8=seq8+list(np.arange(1,181)*1+seq8[-1])         #180 180 180
seq8=seq8+list(np.arange(1,21)*7+seq8[-1])          #2340 180 20


seq9=[]                                             #1 s/scan
seq9.append(1)                                      #0 180 1
seq9=seq9+list(np.arange(1,101)*6+seq9[-1])         #1980 180 100
seq9=seq9+list(np.arange(1,121)*1+seq9[-1])         #180 180 120
seq9=seq9+list(np.arange(1,141)*7+seq9[-1])         #2340 180 140


seq10=[]                                            #0.4 s/scan
seq10.append(f2)                                    #0 180 1
seq10=seq10+list(np.arange(1,181)*1*f2+seq10[-1])   #180 180 180
seq10=seq10+list(np.arange(1,91)*2*f2+seq10[-1])    #540 180 90



TIME={}
TIME['T3_100_6']=seq1
TIME['T3_025_7']=seq1
TIME['T3_100_1']=seq1
TIME['T3_300_5']=seq1
TIME['T3_025_1']=seq1
TIME['T3_025_9']=seq1
TIME['T3_300_4']=seq1
TIME['T3_300_15']=seq1
TIME['T3_100_7']=seq1
TIME['T3_100_10']=seq1

TIME['T4_300_1']=seq2
TIME['T4_025_4']=seq2
TIME['T4_100_5']=seq2
TIME['T4_300_5']=seq2
TIME['T4_100_4']=seq2
TIME['T4_100_3']=seq2
TIME['T4_025_3']=seq2
TIME['T4_025_1']=seq2

TIME['T3_100_4']=seq3

TIME['T3_300_6']=seq4

TIME['T3_300_3']=seq5
TIME['T3_025_4']=seq5
TIME['T3_300_8']=seq5
TIME['T3_025_7_II']=seq5

TIME['T4_300_2_II']=seq6

TIME['T4_025_2_II']=seq7
TIME['T4_300_3_II']=seq7

TIME['T3_025_3_II']=seq8

TIME['T3_300_5_III']=seq9
TIME['T3_100_7_III']=seq9
TIME['T3_100_10_III']=seq9
TIME['T3_025_9_III']=seq9
TIME['T3_300_9_III']=seq9
TIME['T3_100_4_III']=seq9[1:]      #first scan omitted bc of artefacts
TIME['T3_300_3_III']=seq9[1:]        #first scan omitted bc of artefacts
TIME['T3_025_4_III']=seq9
TIME['T3_300_8_III']=seq9
TIME['T3_025_3_III']=seq9

TIME['T4_300_5_III']=seq10
TIME['T4_025_3_III']=seq10
TIME['T4_025_1_III']=seq10
TIME['T4_300_4_III']=seq10
TIME['T4_025_5_III']=seq10
TIME['T4_100_2_III']=seq10
TIME['T4_300_3_III']=seq10
TIME['T4_025_2_III']=seq10
TIME['T4_300_2_III']=seq10



