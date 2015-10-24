from pandas import *  
import glob
from matplotlib import pyplot as plt
import math
import os
import re

# USEFUL RESOURCES
# Split a string with regular expressions http://stackoverflow.com/questions/12683201/python-re-split-to-split-by-spaces-commas-and-periods-but-not-in-cases-like
# Match part of a string as named variable http://stackoverflow.com/questions/1800817/how-can-i-get-part-of-regex-match-as-a-variable-in-python
# Replace character in a string with re http://stackoverflow.com/questions/5658369/how-to-input-a-regex-in-string-replace-in-python

def ensure_dirs_exist(f):
    #  CHECK IF DIRRECTORIES IN FILE PATH EXIST. IF NOT, CREATE THEM.
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

dataLists = {} 

# READ DATA  
for folder in glob.glob("Data/*"):  
    dataLists[folder.split("/")[1]] = []  
      
    for datafile in glob.glob(folder + "/*.csv"):  
        dataLists[folder.split("/")[1]].append(read_csv(datafile, skiprows=[11]))  
  
# CALCULATE STATS FOR DATA
meanDFs = {}  
stderrDFs = {}  

for key in dataLists.keys():    
    keyDF = (concat(dataLists[key], axis=1, keys=range(len(dataLists[key])))  
            .swaplevel(0, 1, axis=1)  
            .sortlevel(axis=1)  
            .groupby(level=0, axis=1)) 
    meanDFs[key] = keyDF.mean()  
    stderrDFs[key] = keyDF.std().div(math.sqrt(len(dataLists[key]))).mul(2.0)
    print len(dataLists[key])
    print 'key: %s' %(key) 
    print stderrDFs[key] 
    keyDF = None 

# PLOT SETTINGS
default_font_size = 15
plt.rcParams.update({'font.size': default_font_size})

# The "Tableau 20" colors as RGB
# Visual of the colors: http://doktorandi.andreasherten.de/2015/03/25/tableau-colors-in-root/
tableau20 = [(31, 119, 180),  (174, 199, 232), (255, 127, 14),  (255, 187, 120),  
             (44, 160, 44),   (152, 223, 138), (214, 39, 40),   (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75),   (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34),  (219, 219, 141), (23, 190, 207),  (158, 218, 229)] 

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.) 

plt.figure(figsize=(10, 8)) 

# PLOTTING
for key in meanDFs.keys():  
    s = re.sub('_',' ',key)
    s = re.split('-', s)
    measure = s[0]
    experiment = s[1]
    plt.cla() 

    # Remove the plot frame lines
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    # Ensure that the axis ticks only show up on the bottom and left of the plot
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()    

    # Add horizontal tick lines
    ax.yaxis.grid(True, lw=2.5, alpha=0.2)

    # Remove the tick marks; they are unnecessary with the tick lines
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="on", left="off", right="off", labelleft="on")  

    plt.title(measure + ' vs. ' + experiment)
    # plt.title(key.replace("_", " ").title())   
    plt.ylabel(measure, fontsize = default_font_size)
    plt.xlabel(experiment, fontsize = default_font_size)  
    
    color_count = 7
    for column in meanDFs[key].columns:
        if (column == "Small_World") or (column == "Tree"):
            plt.errorbar(x=meanDFs[key]["Param"], y=meanDFs[key][column], yerr=stderrDFs[key][column], label=column, lw=2.5, color=tableau20[color_count % 20], alpha = 0.8)  
            color_count -= 7
    plt.legend(fontsize = default_font_size)
    # plt.ylim([output_min, output_max])  

    filename = 'Plots/' + str(key) + '.png'
    ensure_dirs_exist(filename)
    plt.savefig(filename)
