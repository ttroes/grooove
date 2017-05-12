import numpy as np
import scipy
import matplotlib.pyplot as plt
import IPython.display
import essentia
import essentia.standard as es
import librosa
import librosa.display
import pandas as pd
from collections import Counter
import os
import pandas as pd

#input needs to be numpy.ndarray
def nearestNeighbour(targetValue, vector):
    minIndex = 0
    minDiff = targetValue - vector[0]
    minAbsDiff = abs(minDiff)

    for index, value in enumerate(vector):
        diff = targetValue - value
        absDiff = abs(diff)
        if absDiff < minAbsDiff:
            minIndex = index
            minDiff = diff
            minAbsDiff = absDiff
    minDiff = "%.3f" %minDiff #reduce to three decimals
    minIndex= 1 + minIndex %16 #to reduce all deviations into one bar
    return float(minIndex), float(minDiff)

def split_lists(devs):
    ids_one = [item for item in devs if item[0] == 1]
    ids_two = [item for item in devs if item[0] == 2]
    ids_three = [item for item in devs if item[0] == 3]
    ids_four = [item for item in devs if item[0] == 4]
    ids_five = [item for item in devs if item[0] == 5]
    ids_six = [item for item in devs if item[0] == 6]
    ids_seven = [item for item in devs if item[0] == 7]
    ids_eight = [item for item in devs if item[0] == 8]
    ids_nine = [item for item in devs if item[0] == 9]
    ids_ten = [item for item in devs if item[0] == 10]
    ids_eleven = [item for item in devs if item[0] == 11]
    ids_twelve = [item for item in devs if item[0] == 12]
    ids_thirteen = [item for item in devs if item[0] == 13]
    ids_fourteen = [item for item in devs if item[0] == 14]
    ids_fifteen = [item for item in devs if item[0] == 15]
    ids_sixteen = [item for item in devs if item[0] == 16]

def count_instances(ids_one, ids_two, ids_three, ids_four, ids_five, ids_six, ids_seven, ids_eight, ids_nine, ids_ten, ids_eleven, ids_twelve, ids_thirteen, ids_fourteen, ids_fifteen, ids_sixteen):
    c_one = Counter(elem[1] for elem in ids_one)
    c_two = Counter(elem[1] for elem in ids_two)
    c_three = Counter(elem[1] for elem in ids_three)
    c_four = Counter(elem[1] for elem in ids_four)
    c_five = Counter(elem[1] for elem in ids_five)
    c_six = Counter(elem[1] for elem in ids_six)
    c_seven = Counter(elem[1] for elem in ids_seven)
    c_eight = Counter(elem[1] for elem in ids_eight)
    c_nine = Counter(elem[1] for elem in ids_nine)
    c_ten = Counter(elem[1] for elem in ids_ten)
    c_eleven = Counter(elem[1] for elem in ids_eleven)
    c_twelve = Counter(elem[1] for elem in ids_twelve)
    c_thirteen = Counter(elem[1] for elem in ids_thirteen)
    c_fourteen = Counter(elem[1] for elem in ids_fourteen)
    c_fifteen = Counter(elem[1] for elem in ids_fifteen)
    c_sixteen = Counter(elem[1] for elem in ids_sixteen)

def plot_histo_outlier(var1, var2, var3, var4):
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.45, 0.2)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #1')
    plt.hist(var1, 25);
    plt.subplot(1,4,2)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.45, 0.2)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #2')
    plt.hist(var2, 25);
    plt.subplot(1,4,3)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.45, 0.2)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #3')
    plt.hist(var3, 25);
    plt.subplot(1,4,4)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.45, 0.2)
    plt.xlabel('time in seconds')
    plt.title('ideal sixteenth #4')
    plt.hist(var4, 25);

def plot_histogramm1(var1, var2, var3, var4):
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,4,1)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #1')
    plt.hist(var1, 25);
    plt.subplot(1,4,2)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #2')
    plt.hist(var2, 25);
    plt.subplot(1,4,3)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #3')
    plt.hist(var3, 25);
    plt.subplot(1,4,4)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.title('ideal sixteenth #4')
    plt.hist(var4, 25);
    
def plot_histogramm2(var1, var2, var3, var4):
    plt.figure(figsize=(8,4))
    plt.subplot(1,4,1)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #5')
    plt.hist(var1, 25);
    plt.subplot(1,4,2)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #6')
    plt.hist(var2, 25);
    plt.subplot(1,4,3)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #7')
    plt.hist(var3, 25);
    plt.subplot(1,4,4)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.title('ideal sixteenth #8')
    plt.hist(var4, 25);

def plot_histogramm3(var1, var2, var3, var4):
    plt.figure(figsize=(8,4))
    plt.subplot(1,4,1)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #9')
    plt.hist(var1, 25);
    plt.subplot(1,4,2)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #10')
    plt.hist(var2, 25);
    plt.subplot(1,4,3)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #11')
    plt.hist(var3, 25);
    plt.subplot(1,4,4)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.title('ideal sixteenth #12')
    plt.hist(var4, 25);

def plot_histogramm4(var1, var2, var3, var4):
    plt.figure(figsize=(8,4))
    plt.subplot(1,4,1)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #13')
    plt.hist(var1, 25);
    plt.subplot(1,4,2)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #14')
    plt.hist(var2, 25);
    plt.subplot(1,4,3)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.ylabel('counts')
    plt.title('ideal sixteenth #15')
    plt.hist(var3, 25);
    plt.subplot(1,4,4)
    plt.axvline(x=0, color='y')
    plt.xlim(-0.1, 0.1)
    plt.xlabel('time in seconds')
    plt.title('ideal sixteenth #16')
    plt.hist(var4, 25);

def plot_boxplot1(var1,var2,var3,var4):
    variables = [var1,var2,var3,var4]
    plt.axhline(y=0, color='g', linestyle = 'dashdot')
    plt.title('first beat')
    plt.boxplot(variables, 0, ''); # delete outliers
def plot_boxplot2(var1,var2,var3,var4):
    variables = [var1,var2,var3,var4]
    plt.axhline(y=0, color='g', linestyle = 'dashdot')
    plt.title('second beat')
    plt.boxplot(variables, 0, ''); # delete outliers
def plot_boxplot3(var1,var2,var3,var4):
    variables = [var1,var2,var3,var4]
    plt.axhline(y=0, color='g', linestyle = 'dashdot')
    plt.title('third beat')
    plt.boxplot(variables, 0, ''); # delete outliers
def plot_boxplot4(var1,var2,var3,var4):
    variables = [var1,var2,var3,var4]
    plt.axhline(y=0, color='g', linestyle = 'dashdot')
    plt.title('fourth beat')
    plt.boxplot(variables, 0, ''); # delete outliers
