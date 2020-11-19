# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:58:06 2020

@author: krish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import automation as am
plt.style.use('science')

#----------------------------- Import Excel ----------------------------------#

analysis = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Nanoindentation\Results - Indentation test - 1\sample1_shale_modified.xls', sheet_name = 'Analysis')
test = pd.DataFrame(analysis,columns= ['Test'])
test = test.to_numpy()
test = test[1:]
test = test.astype(np.float)
avgmod = pd.DataFrame(analysis,columns= ['Avg Modulus [3000-4000 nm]'])
avgmod = avgmod.to_numpy()
avgmod = avgmod[1:]
avgmod = avgmod.astype(np.float)
avghard = pd.DataFrame(analysis,columns= ['Avg Hardness [3000-4000 nm]'])
avghard = avghard.to_numpy()
avghard = avghard[1:]
avghard = avghard.astype(np.float)

#-----------------------------------------------------------------------------#


#-------------------------- Modulus - Histogram ------------------------------#

mu_mod = np.mean(avgmod)
sigma_mod = np.std(avgmod)
x = np.linspace(np.min(avgmod),np.max(avgmod), 100)
y = norm.pdf(x, loc = mu_mod, scale = sigma_mod)
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.hist(avgmod, density=True, bins=30, label="Data", color='blue')
plt.plot(x,y, linestyle = '-',color = 'black', linewidth=1,label="PDF")
plt.legend(loc='best', prop={'size': 5})
plt.autoscale(tight=True)
plt.xlabel('Avg Modulus in depth range of 3000-4000 nm [GPa]')
plt.ylabel('Probability')
plt.title('Avg Modulus Histogram from nanoindentation',fontsize=10)
plt.grid(True)
plt.savefig('Histogram_Modulus.png',dpi=600)

#-----------------------------------------------------------------------------#


#-------------------------- Hardness - Histogram -----------------------------#

mu_har = np.mean(avghard)
sigma_har = np.std(avghard)
x1 = np.linspace(np.min(avghard),np.max(avghard), 100)
y1 = norm.pdf(x1, loc = mu_har, scale = sigma_har)
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.hist(avghard, density=True, bins=30, label="Data", color='green')
plt.plot(x1,y1, linestyle = '-',color = 'black', linewidth=1,label="PDF")
plt.legend(loc='best', prop={'size': 5})
plt.autoscale(tight=True)
plt.xlabel('Avg Hardness in depth range of 3000-4000 nm [GPa]', fontsize=8)
plt.ylabel('Probability', fontsize=8)
plt.title('Avg Hardness Histogram from nanoindentation',fontsize=10)
plt.grid(True)
plt.savefig('Histogram_Hardness.png',dpi=600)

#-----------------------------------------------------------------------------#
 

#----------------------------- Import Excel ----------------------------------#

analysis = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Nanoindentation\Results - Indentation test - 1\sample1_shale_modified.xls', sheet_name = 'Analysis')
test = pd.DataFrame(analysis,columns= ['Test'])
test = test.stack().tolist()
test = test[1:]
avgmod = pd.DataFrame(analysis,columns= ['Avg Modulus [3000-4000 nm]'])
avgmod = avgmod.stack().tolist()
avgmod = avgmod[1:]

avghard = pd.DataFrame(analysis,columns= ['Avg Hardness [3000-4000 nm]'])
avghard = avghard.stack().tolist()
avghard = avghard[1:]

#-----------------------------------------------------------------------------#

maxmod = max(avgmod)
minmod = min(avgmod)
maxindent = test[avgmod.index(maxmod)].zfill(3)
minindent = test[avgmod.index(minmod)].zfill(3)


n = maxindent  # Indent Number with maximum load
n1 = minindent  # Indent Number with minimum load
n2 = str(5).zfill(3)
disp, load, hcf, modulus, hardness = am.excelprocess(n,100)
disp1, load1, hcf1, modulus1, hardness1 = am.excelprocess(n1,100) 
disp2, load2, hcf2, modulus2, hardness2 = am.excelprocess(n2,100) 

#-----------------------------------------------------------------------------#


#---------------------- Load vs Displacement graph ---------------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,load, 'r.', alpha = 0.5, markersize=1)
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Load on Sample [mN]', fontsize=8)
plt.title('Load vs Displacement curve of indent - '+n,fontsize=10)
plt.autoscale()
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.savefig('Load vs Displacement - single indent.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#


#------------------ Load vs Displacement comparison graph --------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,load, 'b.', alpha = 0.5, markersize=1,label='Indent '+n)
plt.plot(disp1,load1, 'r.', alpha = 0.5, markersize=1,label='Indent '+n1)
plt.plot(disp2,load2, 'g.', alpha = 0.5, markersize=1,label='Indent '+n2)
plt.axvline(x=5000, color = 'k', linewidth=0.3, label = 'maximum depth')
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Load on Sample [mN]', fontsize=8)
plt.title('Load vs Displacement comparison of different indents',fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.autoscale()
plt.grid(True)
plt.legend(loc='best', prop={'size': 5})
plt.savefig('Load vs Displacement - indent comparison.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#


#------------ Harmonic Contact stiffness vs Displacement graph ---------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,hcf, 'b.', alpha = 0.5, markersize=1)
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Harmonic Contact Stiffness [N/m]', fontsize=8)
plt.title('Harmonic Contact Stiffness vs Displacement curve of indent - '+n,fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Harmonic Contact Stiffness vs Displacement - single indent.png', dpi = 600, quality = 100,orientation='landscape')


#-----------------------------------------------------------------------------#


#------ Harmonic Contact stiffness vs Displacement comparison graph ----------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,hcf, 'b.', alpha = 0.5, markersize=1,label='Indent '+n)
plt.plot(disp1,hcf1, 'r.', alpha = 0.5, markersize=1,label='Indent '+n1)
plt.plot(disp2,hcf2, 'g.', alpha = 0.5, markersize=1,label='Indent '+n2)
plt.axvline(x=5000, color = 'k', linewidth=0.3, label = 'maximum depth')
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Harmonic Contact Stiffness [N/m]', fontsize=8)
plt.title('Harmonic Contact Stiffness vs Displacement comparison of different indents',fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.legend(loc='lower right', prop={'size': 5})
plt.autoscale()
plt.grid(True)
plt.savefig('Harmonic Contact Stiffness vs Displacement - indent comparison.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#


#--------------------- Modulus vs Displacement graph -------------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,modulus, 'g.', alpha = 0.5, markersize=1)
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Modulus [GPa]', fontsize=8)
plt.title('Modulus vs Displacement curve of indent - '+n,fontsize=8)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Modulus vs Displacement - single indent.png', dpi = 600, quality = 100,orientation='landscape')


#-----------------------------------------------------------------------------#

#---------------- Modulus vs Displacement comparison graph -------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,modulus, 'b.', alpha = 0.5, markersize=1,label='Indent '+n)
plt.plot(disp1,modulus1, 'r.', alpha = 0.5, markersize=1,label='Indent '+n1)
plt.plot(disp2,modulus2, 'g.', alpha = 0.5, markersize=1,label='Indent '+n2)
plt.axvline(x=5000, color = 'k', linewidth=0.3, label = 'maximum depth')
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Modulus [GPa]', fontsize=8)
plt.title('Modulus vs Displacement comparison of different indents',fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.legend(loc='best', prop={'size': 5})
plt.autoscale()
plt.grid(True)
plt.savefig('Modulus vs Displacement - indent comparison.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#


#--------------------- Hardness vs Displacement graph ------------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,hardness, 'm.', alpha = 0.5, markersize=1)
plt.xlabel('Displacement into Surface [nm]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
plt.title('Hardness vs Displacement curve of indent - '+n,fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Hardness vs Displacement - single indent.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#

#---------------- Hardness vs Displacement comparison graph ------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(disp,hardness, 'b.', alpha = 0.5, markersize=1,label='Indent '+n)
plt.plot(disp1,hardness1, 'r.', alpha = 0.5, markersize=1,label='Indent '+n1)
plt.plot(disp2,hardness2, 'g.', alpha = 0.5, markersize=1,label='Indent '+n2)
plt.axvline(x=5000, color = 'k', linewidth=0.3, label = 'maximum depth')
plt.xlabel('Displacement into Surface [nanometers - nm]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
plt.title('Hardness vs Displacement comparison of different indents',fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.legend(loc='best', prop={'size': 5})
plt.autoscale()
plt.grid(True)
plt.savefig('Hardness vs Displacement - indent comparison.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#

#----------------------- Greyscale - Histogram -------------------------------#

greyscale = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Lab Data\MicroCT\Histogram of shalepol10890.xls', sheet_name = 'Histogram of shalepol10890')
index = pd.DataFrame(greyscale,columns= ['index'])
index = index.stack().tolist()
index = index[1:]
count = pd.DataFrame(greyscale,columns= ['count'])
count = count.stack().tolist()
count = count[1:]
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.bar(index,count)
plt.xlabel('Bins - Intensity')
plt.ylabel('Greyscale count')
plt.title('Greyscale Histogram of the RVE from microCT',fontsize=10)
plt.grid(True)
plt.autoscale()
plt.savefig('Histogram_Greyscale.png',dpi=600)

#-----------------------------------------------------------------------------#

#----------------------- Modulus versus Hardness -----------------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
iters = 0
while iters<24:
    number = test[iters].zfill(3)
    disp3, load3, hcf3, modulus3, hardness3 = am.excelprocess(number,150) 
    plt.plot(modulus3,hardness3, 'k.', alpha = 0.5, markersize=1)
    iters = iters+1
    print(iters)
    print(number)
    
plt.xlabel('Modulus [GPa]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
plt.title('Modulus vs Hardness',fontsize=10)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Modulus vs Hardness.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#