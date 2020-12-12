# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:31:27 2020

@author: krish
"""
import pandas as pd


def excelprocess(n,exclude):
    sheetname = 'Test ' + n
    indents = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Nanoindentation\Results - Indentation test - 1\sample1_shale_modified.xls', sheet_name = sheetname)
    disp = pd.DataFrame(indents,columns= ['Displacement Into Surface'])
    load = pd.DataFrame(indents,columns= ['Load On Sample'])
    hcf = pd.DataFrame(indents,columns= ['Harmonic Contact Stiffness'])
    modulus = pd.DataFrame(indents,columns= ['Modulus'])
    hardness = pd.DataFrame(indents,columns= ['Hardness'])
    disp = disp.stack().tolist()
    disp = disp[exclude:]
    load = load.stack().tolist()
    load = load[exclude:]
    hcf = hcf.stack().tolist()
    hcf = hcf[exclude:]
    modulus = modulus.stack().tolist()
    modulus = modulus[exclude:]
    hardness = hardness.stack().tolist()
    hardness = hardness[exclude:]
    return disp, load, hcf, modulus, hardness