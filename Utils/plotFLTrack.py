from Utils.helpers import smooth

import numpy as np
import matplotlib.pyplot as plt

def plotFlTrack(tl):
    total = 0
    plt.figure()
    for i in range(tl.total_cells):
        x, gfpfl, rfpfl = tl[i]
        if len(x)>30 and total<15 and i != 5:
            total+=1
            diff = int(abs(len(x) - len(gfpfl)) / 2)
            ratio = np.array(rfpfl) / np.array(gfpfl) 
            ratio = smooth(ratio, window_len=5)
            diff = int(abs(len(x) - len(ratio)) / 2)
            ratio = ratio[diff:-diff]
            plt.plot(np.array(x) * 10,ratio, 'k')

    plt.ylabel('mCherry/sfGFP Fluorescence Ratio', fontsize=15)
    plt.xlabel('Time (minutes)', fontsize=15)
    plt.show()