## Common imports
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import os
import seaborn as sns
import pylab as plt

import statsmodels.formula.api as smf
log = np.log #fix statsmodels

from scipy.stats import binom 

from adjustText import adjust_text


# # import these modules
# import nltk
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()




# Visualization colors
cond2color = {"Chain":"#e6af2e", "Network":"#3d348b"}

conditions = ["Chain","Network"]



## Visualizations

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, "axes.spines.bottom":
    False,"lines.linewidth": 2, "grid.color": "lightgray", "legend.frameon": False, "xtick.labelcolor": "#484848", "ytick.labelcolor":
    "#484848", "xtick.color": "#484848", "ytick.color": "#484848","text.color": "#484848", "axes.labelcolor": "#484848",
    "axes.titlecolor":"#484848","figure.figsize": [5,3],
    "axes.titlelocation":"left","xaxis.labellocation":"left","yaxis.labellocation":"bottom"}
palette = ["#3d348b","#e6af2e","#191716","#e0e2db"] #use your favourite colours
sns.set_theme(context='paper', style='white', palette=palette, font='Verdana', font_scale=1.3, color_codes=True,
rc=custom_params)

def plot_agg(data, x, y, hue, cond2color={"Chain":"#e6af2e", "Network":"#3d348b"}):
    """
    Plots individual traces for each replication. In the network condition, it averages the three people in the layer.

    Parameters:
    - data (DataFrame): The input DataFrame containing the data to be plotted.
    - x (str): The column name in 'data' to be used as the x-axis variable for plotting.
    - y (str): The column name in 'data' to be used as the y-axis variable for plotting.
    - hue (str): The column name in 'data' used to determine the color and style of the plot lines based on unique values.
    - cond2color (dict): The matching between condition name and color
    
    Returns:
    None. This function directly generates a plot without returning any values.    
    """
    
    for cond, dt in data.groupby(hue):  # Group by condition
        for rep, dt_rep in dt.groupby("rep"):  # Further group by replication factor
            if cond == "Network":
                # For "Network" condition, average the data grouped by the x-axis variable
                dt_rep = dt_rep.groupby(x).mean(numeric_only=True).reset_index()
                
            # Plot the data with style and opacity settings based on condition
            plt.plot(dt_rep[x], dt_rep[y], "--", color=cond2color[cond], alpha=0.2)


def set_style(name, xlabel="Generation number"):
    """
    Configures the style of matplotlib plots with specified labels and aesthetics.

    Parameters:
    - name (str): The label for the y-axis.
    - xlabel (str, optional): The label for the x-axis. Defaults to "Generation number".

    Returns:
    None. This function directly modifies the matplotlib plotting configuration.
    """
    plt.xlabel(xlabel)
    plt.ylabel(name)
    plt.grid("on")
    plt.rc('axes', linewidth=2)
    plt.gca().grid(axis='x')
    
    sns.despine(left=True, bottom=True)  # Removes the top and right border for a cleaner look
    plt.legend(loc=1, frameon=False, ncol=2)  # Adjusts legend positioning and appearance

def plot_similarity_indep_replicates(r, ax=None):
    sns.lineplot(x="layer_n",y="story_merged",hue="condition",
             palette=cond2color, data=r,lw=2,marker="o",markersize=7)
    
    set_style("Similarity between independent stories")

def find_color(title):
    if "Network" in title:
        color = cond2color["Network"]
    elif "Chain" in title:
        color = cond2color["Chain"]
    else:
        color = "gray"
    return color


def plot_embedded(X_embedded,data,filter_,title="Network",traces=True):
    xmin,ymin,xmax,ymax = np.concatenate([X_embedded.min(0)-0.05,X_embedded.max(0)+0.05])
    values = sns.color_palette("rocket_r",7)
    colors_reps = sns.color_palette("tab10",len(data.loc[filter_,"rep"].unique()))

    if traces:
        for rep in sorted(data.loc[filter_,"rep"].unique()):
            filter_2 = filter_ & (data["rep"]==rep)
            plt.plot(np.concatenate([[X_embedded[-1,0]],data.loc[filter_2,"x"].values]), 
                     np.concatenate([[X_embedded[-1,1]],data.loc[filter_2,"y"].values]),
                     color=colors_reps.pop(0),zorder=0,alpha=0.5)

    
    plt.scatter(data.loc[filter_,"x"], data.loc[filter_,"y"],color=[values[_] for _ in data.loc[filter_,"layer_n"]])
    plt.scatter(X_embedded[-1,0], X_embedded[-1,1], s=150, color = "cornflowerblue")

    sns.despine(bottom=True,left=True) 
    color = find_color(title)
    plt.title(title, color=color)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    # Remove axis ticks
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labelleft=False)
    
def plot_transmissions(data2, X_embedded, traces=True):
    plt.figure(figsize=(10,4), facecolor="white")
    plt.subplot(121, facecolor='.95')
    filter_ = data2["condition"] == "Network"
    plot_embedded(X_embedded,data2,filter_,"(A) Network",traces=traces)

    plt.subplot(122, facecolor='.95')
    filter_ = data2["condition"] == "Chain"
    plot_embedded(X_embedded,data2,filter_,"(B) Chain",traces=traces)

    plt.tight_layout()

    


