import math
import os
import sys
from os import path
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
from matplotlib.pyplot import figure
import glob
import pandas as pd
figure(figsize=(8, 8), dpi=600)



def plot_ROC(label, pred, lineColor, markerStyle, method):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - label (array-like): True binary labels.
    - pred (array-like): Target scores or probability estimates.
    - lineColor (str): Color of the ROC curve.
    - markerStyle (str): Marker style for data points on the ROC curve.
    - method (str): Label for the method being plotted.

    Returns:
    None
    """
    fpr, tpr, thresholds = roc_curve(label, pred)
    sns.lineplot(x=fpr, y=tpr, marker=markerStyle, color=lineColor, markevery=(0.1, 0.1), label=method)#mec='#1D6996', markerfacecolor='#1D6996',mew=0.3, markersize=15, xlabel='common xlabel', ylabel='common ylabel')

def plot_PRC(label, pred, lineColor, markerStyle, method):
    """
    Plots the Precision-Recall Curve (PRC).

    Parameters:
    - label (array-like): True binary labels.
    - pred (array-like): Target scores or probability estimates.
    - lineColor (str): Color of the PRC curve.
    - markerStyle (str): Marker style for data points on the PRC curve.
    - method (str): Label for the method being plotted.

    Returns:
    None
    """
    precision, recall, thresholds = precision_recall_curve(label, pred)
    sns.lineplot(x=recall, y=precision, marker=markerStyle, color=lineColor, markevery=(0.1, 0.1), label=method)#mec='#1D6996', markerfacecolor='#1D6996',mew=0.3, markersize=15, xlabel='common xlabel', ylabel='common ylabel')




        


def ploter(plot_method, dataset):
    """
    Plots either ROC or Precision-Recall Curve (PRC) for multiple methods on a given dataset.

    Parameters:
    - plot_method (str): Specifies whether to plot 'ROC' or 'PRC'.
    - dataset (str): Path to the directory containing method files.

    Returns:
    None
    """
    lgndOrder = dict()
    cnt = 0
    for mthdFile in glob.glob('{}/*'.format(dataset)):
        mthdName = mthdFile.split('.')[0].split('/')[-1]
        lgndOrder[mthdName] = cnt
        df = pd.read_csv(mthdFile, sep='\t')
        label, pred = df.label, df.prediction
        if plot_method == 'ROC':
            plot_ROC(label, pred, lineColor[mthdName], markerStyle[mthdName], mthdName)
            
        else:
            plot_PRC(label, pred, lineColor[mthdName], markerStyle[mthdName], mthdName)
        cnt+=1
        if cnt ==20:
            break
    order = list(lgndOrder.values())
    order.remove(lgndOrder['Ensemble+puzzler'])
    order.append(lgndOrder['Ensemble+puzzler'])
    handles, labels = plt.gca().get_legend_handles_labels()

    if plot_method == 'ROC':
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right') 
        plt.title('ROC Curve {}'.format(dataset.split('/')[-1]))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig("plots/ROC_{}.pdf".format(dataset.split('/')[-1]))
    else:
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right') 
        plt.title('PR Curve {}'.format(dataset.split('/')[-1]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig("plots/PRC_{}.pdf".format(dataset.split('/')[-1]))
    plt.clf()



def method_ploter(plot_method, methods, dataset):
    """
    Plots either ROC or Precision-Recall Curve (PRC) for multiple methods on a given dataset.

    Parameters:
    - plot_method (str): Specifies whether to plot 'ROC' or 'PRC'.
    - dataset (str): Path to the directory containing method files.

    Returns:
    None
    """
    speciesDict = {'ecoli':'E.coli', 'mouse':'M.musculus', 'worm':'C.elegans', 
                   'fly':'D.melanogaster', 'yeast':'S.cerevisiae '}
    lgndOrder = dict()
    cnt = 0
    for mthdName in methods:
        lgndOrder[mthdName] = cnt
        df = pd.read_csv(dataset+mthdName+'.tsv', sep='\t')
        label, pred = df.label, df.prediction
        if plot_method == 'ROC':
            plot_ROC(label, pred, lineColor[mthdName], markerStyle[mthdName], mthdName)
            
        else:
            plot_PRC(label, pred, lineColor[mthdName], markerStyle[mthdName], mthdName)
        cnt+=1
        if cnt ==20:
            break
    order = list(lgndOrder.values())
    order.remove(lgndOrder['C3PI'])
    order.append(lgndOrder['C3PI'])
    handles, labels = plt.gca().get_legend_handles_labels()

    if plot_method == 'ROC':
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right') 
        plt.title('ROC Curve {}'.format(speciesDict[dataset.split('/')[-2]]))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig("plots/ROC_{}_methods_.pdf".format(dataset.split('/')[-2]))#, bbox_inches='tight')
    else:
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right') 
        plt.title('PR Curve {}'.format(speciesDict[dataset.split('/')[-2]]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig("plots/PRC_{}_methods_.pdf".format(dataset.split('/')[-2]))#, bbox_inches='tight')
    plt.clf()


lineColor = {'D-SCRIPT':'#1b9e77', 'CNN1D':'#d95f02', 'CNN2D':'#7570b3','CNN1D+puzzler':'#a6761d',
             'CNN2D+puzzler':'#66a61e','Ensemble':'#e6ab02','Ensemble+puzzler':'red'}
markerStyle = {'D-SCRIPT':'o', 'CNN1D':'s','CNN2D':'p', 'CNN1D+puzzler':'v', 
               'CNN2D+puzzler':11, 'Ensemble':'d', 'Ensemble+puzzler':'X'} 

lineColor = {'D-SCRIPT':'red' , 'Topsy-Turvy':'blue', 'C3PI':'green'}
markerStyle = {'D-SCRIPT':'o', 'Topsy-Turvy':'s', 'C3PI':'X'}


methods = ['D-SCRIPT', 'Topsy-Turvy', 'C3PI']

print("ecoli")
method_ploter('PRC', methods, "predictions/ecoli/")
method_ploter('ROC', methods, "predictions/ecoli/")

print("worm")
method_ploter('PRC', methods, "predictions/worm/")
method_ploter('ROC', methods, "predictions/worm/")

print("mouse")
method_ploter('PRC', methods, "predictions/mouse/")
method_ploter('ROC', methods, "predictions/mouse/")

print("yeast")
method_ploter('PRC', methods, "predictions/yeast/")
method_ploter('ROC', methods, "predictions/yeast/")

print("fly")
method_ploter('PRC', methods, "predictions/fly/")
method_ploter('ROC', methods, "predictions/fly/")


"""
print("ecoli")
ploter('PRC', "predictions/ecoli")
ploter('ROC', "predictions/ecoli")

print("worm")
ploter('PRC', "predictions/worm")
ploter('ROC', "predictions/worm")

print("mouse")
ploter('PRC', "predictions/mouse")
ploter('ROC', "predictions/mouse")

print("yeast")
ploter('PRC', "predictions/yeast")
ploter('ROC', "predictions/yeast")

print("fly")
ploter('PRC', "predictions/fly")
ploter('ROC', "predictions/fly")
"""
#print("human")
#ploter('PRC', "predictions/human")
#ploter('ROC', "predictions/human")
