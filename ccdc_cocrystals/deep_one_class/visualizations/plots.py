import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mpld3
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.lines import Line2D
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams["font.weight"] = "light"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 16})

def ranking_plot(scores, true_label,names, save_dir=''):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set(adjustable='box')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis = 'both', which='both', width=3)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis = 'both', which='major', length=8)
    ax.tick_params(axis = 'both', which='both' , bottom=True, top=True, left=True, right=True, direction='in')
    ax.axis('on')
    ax.grid(False)
    ax.set_facecolor('xkcd:white')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('score', fontsize=16)
    plt.xlabel('rank', fontsize=16)
    labels = true_label
    pub_validation=pd.concat([pd.DataFrame(scores, columns=['score']), pd.DataFrame(labels.values, columns=['color'])], axis=1)
    pub_val_sort = pub_validation.sort_values(by='score', ascending=False) 
    x=np.arange(len(scores))
    ax.scatter(x ,pub_val_sort.score, c=pub_val_sort.color)
    plt.title('Validation dataset')
    plt.title('Validation dataset')
    plt.savefig(f'{save_dir}/ranking_plot.png', dpi=600)
    plt.show()

def plot_confusion_matrix(true_label, scores, threshold, save_dir=''):
    fig, ax = plt.subplots(figsize=(6,5))    
    ax.set_xlabel({'size':'16'} )
    ax.set_ylabel({'size':'16'} )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    y_pred  =[1 if i>=threshold else 0 for i in scores]
    tn, fp, fn, tp = confusion_matrix(true_label, y_pred).ravel()
    specificity = tn / (tn+fp)
    roc_auc = roc_auc_score(true_label, scores)
    recall=recall_score(true_label, y_pred)
    ax.set_title('Specificity : ' + str(round(specificity,2)) +
             '\n Roc_auc_score : '+ str(round(roc_auc,2)) +
             '\n Recall_score : ' +  str(round(recall,2)) )
    cm = confusion_matrix(true_label, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=(0,1)).plot(ax=ax)
    plt.savefig(f'{save_dir}/confusion_matrix.png', bbox_inches='tight' ,dpi=600)
    plt.show()