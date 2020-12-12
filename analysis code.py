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
from sklearn.metrics import r2_score
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
cov_mod = mu_mod/sigma_mod
x = np.linspace(np.min(avgmod),np.max(avgmod), 100)
y = norm.pdf(x, loc = mu_mod, scale = sigma_mod)
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.hist(avgmod, density=True, bins=30, label="Data", color='blue')
plt.plot(x,y, linestyle = '-',color = 'black', linewidth=1,label="PDF")
plt.legend(loc='best', prop={'size': 5})
plt.autoscale(tight=True)
plt.xlabel('Modulus in depth range of 3000-4000 nm [GPa]')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('Histogram_Modulus.png',dpi=600)


#-----------------------------------------------------------------------------#


#-------------------------- Hardness - Histogram -----------------------------#

mu_har = np.mean(avghard)
sigma_har = np.std(avghard)
cov_har = mu_har/sigma_har
x1 = np.linspace(np.min(avghard),np.max(avghard), 100)
y1 = norm.pdf(x1, loc = mu_har, scale = sigma_har)
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.hist(avghard, density=True, bins=30, label="Data", color='green')
plt.plot(x1,y1, linestyle = '-',color = 'black', linewidth=1,label="PDF")
plt.legend(loc='best', prop={'size': 5})
plt.autoscale(tight=True)
plt.xlabel('Hardness in depth range of 3000-4000 nm [GPa]', fontsize=8)
plt.ylabel('Probability', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Load on sample [mN]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Load on sample [mN]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Harmonic contact stiffness [N/m]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Harmonic contact stiffness [N/m]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Modulus [GPa]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Modulus [GPa]', fontsize=8)
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
plt.xlabel('Displacement into surface [nm]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
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
plt.xlabel('Displacement into surface [nanometers - nm]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.legend(loc='best', prop={'size': 5})
plt.autoscale()
plt.grid(True)
plt.savefig('Hardness vs Displacement - indent comparison.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#

#----------------------- Modulus versus Hardness -----------------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(avgmod,avghard, 'k.', alpha = 0.5, markersize=1)

z = np.polyfit(avgmod, avghard, 1)
y_hat = np.poly1d(z)(avgmod)

plt.plot(avgmod, y_hat, "r--", lw=0.3)
text = f"$H={z[0]:0.3f}\;M{z[1]:+0.3f}$\n$R^2 = {r2_score(avghard,y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=8, verticalalignment='top')
plt.xlabel('Modulus [GPa]', fontsize=8)
plt.ylabel('Hardness [GPa]', fontsize=8)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.autoscale()
plt.savefig('Modulus vs Hardness.png', dpi = 600, quality = 100,orientation='landscape')

#-----------------------------------------------------------------------------#

#----------------------- Greyscale - Histogram -------------------------------#

greyscale = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Lab Data\MicroCT\Histogram of shalepol10890.xls', sheet_name = 'Histogram of shalepol10890')
index = pd.DataFrame(greyscale,columns= ['index'])
index = index.stack().tolist()
index = index[1:]
count = pd.DataFrame(greyscale,columns= ['count'])
count = count.stack().tolist()

#-----------------------------------------------------------------------------#

#-------------------- Greyscale - Histogram - ESEM ---------------------------#

count = count[1:]
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.bar(index,count)
plt.xlabel('Gray-scale level')
plt.ylabel('Gray-scale count')
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Histogram_Greyscale.png',dpi=600)

#--------------------------------
greyscale = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Histogram_ESEM\Histogram of Shale-1_001 (G).xls', sheet_name = 'Histogram of Shale-1_001 (G)')
value = pd.DataFrame(greyscale,columns= ['value'])
value = value.stack().tolist()
value1 = value[1:]
count = pd.DataFrame(greyscale,columns= ['count'])
count = count.stack().tolist()
count1 = count[1:]
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(value1,count1,color = 'black', linewidth=1)
plt.bar(value1,count1, color='blue')
plt.xlabel('Gray-scale level')
plt.ylabel('Gray-scale count')
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Histogram_Greyscale_ESEM.png',dpi=600)

#-----------------------------------------------------------------------------#

#--------------- Greyscale - Histogram - ESEM Indent -------------------------#

greyscale = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Histogram_ESEM\Histogram of ShaleIndent101_001 (G).xls', sheet_name = 'Histogram of ShaleIndent101_001')
value = pd.DataFrame(greyscale,columns= ['value'])
value = value.stack().tolist()
value2 = value[1:]
count = pd.DataFrame(greyscale,columns= ['count'])
count = count.stack().tolist()
count2 = count[1:]
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(value2,count2,color = 'black', linewidth=1)
plt.bar(value2,count2, color='red')
plt.xlabel('Gray-scale level')
plt.ylabel('Gray-scale count')
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Histogram_Greyscale_ESEM_Indent.png',dpi=600)

#-----------------------------------------------------------------------------#


#------------- Greyscale - Histogram - ESEM comparison -----------------------#

plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(value2,count2,color = 'red', linewidth=1, label='after indentation')
plt.plot(value1,count1,color = 'blue', linewidth=1, label='before indentation')
plt.xlabel('Gray-scale level')
plt.ylabel('Gray-scale count')
plt.grid(True)
plt.autoscale()
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.legend(loc='best', prop={'size': 5})
plt.savefig('Histogram_Greyscale_comparison.png',dpi=600)

#-----------------------------------------------------------------------------#

#------------------ Greyscale - Histogram - microCT --------------------------#

greyscale = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesis\Lab Data\MicroCT\Images\Histogram of shale.xls', sheet_name = 'Histogram of shale')
value = pd.DataFrame(greyscale,columns= ['value'])
value = value.stack().tolist()
value3 = value[1:]
count = pd.DataFrame(greyscale,columns= ['count'])
count = count.stack().tolist()
count3 = count[1:]
plt.rc('font', family='serif')
plt.figure(figsize=(4, 3))
plt.plot(value3,count3,color = 'black', linewidth=1)
plt.bar(value3,count3, color='green')
plt.xlabel('Gray-scale level')
plt.ylabel('Gray-scale count')
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.grid(True)
plt.autoscale()
plt.savefig('Histogram_Greyscale_microCT.png',dpi=600)

#-----------------------------------------------------------------------------#

#--------------------- Modulus 2D Contour Map --------------------------------#

# analysis = pd.read_excel (r'C:\Users\krish\Desktop\TU Delft Files\Additional Thesi  s\Nanoindentation\Results - Indentation test - 1\sample1_shale.xls', sheet_name = 'Analysis')
# test = pd.DataFrame(analysis,columns= ['Test'])
# test = test.to_numpy()
# test = test[1:]
# test = test.astype(np.float)
# avgmod = pd.DataFrame(analysis,columns= ['Avg Modulus [3000-4000 nm]'])
# avgmod = avgmod.to_numpy()
# avgmod = avgmod[1:]
# avgmod = avgmod.astype(np.float)
# avghard = pd.DataFrame(analysis,columns= ['Avg Hardness [3000-4000 nm]'])
# avghard = avghard.to_numpy()
# avghard = avghard[1:]
# avghard = avghard.astype(np.float)

# # gridx = np.linspace(1,25,25)
# # gridy = np.linspace(1,20,20)

# # GX,GY = np.meshgrid(gridx,gridy)

# # fillzero = np.zeros((85,1), dtype=np.float)
# # avgmod = np.append(avgmod,fillzero)
# # m = np.zeros((20,25), dtype=np.float)

# # for i in range(20):
# #     k = 0
# #     for j in range(25):
# #         k = k+j+1
# #         print(k)
# #         m[i][j] = avgmod[j]
 
# # M = avgmod

# # plt.contour(GX,GY,M, colors='black')

#-----------------------------------------------------------------------------#

print('Modulus:')
print('Mean =', mu_mod)
print('Standard deviation =', sigma_mod)
print('CoV =',cov_mod)

print('Hardness:')
print('Mean =', mu_har)
print('Standard deviation =', sigma_har)
print('CoV =',cov_har)

#-----------------------------------------------------------------------------#

