# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:46:37 2020

@author: firo
"""

import os


sourceFolder = r'/scratch/snx3000/fischer/pescatore'
sourceFolder = r"W:\Robert_TOMCAT_3_netcdf4_archives"

destination = os.path.join(sourceFolder, 'test')

if not os.path.exists(destination):
    os.mkdir(destination)
    
content = os.listdir(os.path.join(sourceFolder,'fiber_data'))#, 'fiber_data'))

regfile=os.path.join(destination,'register_log.txt')
reglog=open(regfile,"w")
for item in content:
    print(item)
    reglog.write(item+'\n')
reglog.close() 