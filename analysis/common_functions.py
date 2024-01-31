## Common imports
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import os
import seaborn as sns
import pylab as plt


import string
from collections import Counter

import statsmodels.formula.api as smf
log = np.log #fix statsmodels

from scipy.stats import binom 
from sentence_transformers import SentenceTransformer, util
from adjustText import adjust_text


import itertools
from sklearn.metrics.pairwise import cosine_similarity

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




## Text processing
def story2set(story, create_set=True, stop_words={}):
    """
    Converts a story (text) into a set or list of words, optionally removing stop words and punctuation.

    Parameters:
    - story (str): The text of the story to be processed.
    - create_set (bool, optional): If True, returns a set of words, removing duplicates. If False, returns a list. Defaults to True.
    - stop_words (set, optional): A set of words to be excluded from the output. Defaults to an empty set.

    Returns:
    set or list: Depending on the value of create_set, returns either a set or a list of words from the story.
    """
    if isinstance(story, float):  # Handles NaN values
        return np.NaN
    else:
        table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation
        stripped = [w.translate(table) for w in story.lower().split() if w not in stop_words]  # Removes punctuation and stop words
        stripped = [w for w in stripped if len(w) > 0]  # Removes empty strings
        
        return set(stripped) if create_set else stripped

## Jaccard, Propagation, Inference, Forgetting, Union Functions
# These functions calculate different metrics of similarity or interaction between sets of words, which can be derived from stories or texts. They are particularly useful for analyzing the overlap and unique contributions among different versions or parts of stories.
def jaccard(strs):
    str1,str2 = strs
    if isinstance(str1,float) or isinstance(str2,float):
        return np.NaN
    return len(str1 & str2)/ len(str1 | str2)

def propagation(strs):
    str1,str2,str3 = strs
    if isinstance(str1,float) or isinstance(str2,float)  or isinstance(str3,float):
        return np.NaN
    return len(str1&str2&str3)/len(str1|str2|str3)

def inference(strs):    
    str1,str2,str3 = strs
    if isinstance(str1,float) or isinstance(str2,float)  or isinstance(str3,float):
        return np.NaN
    return len((str3-str1)-str2)/len(str1|str2|str3)


def forgetting(strs):    
    str1,str2,str3 = strs
    if isinstance(str1,float) or isinstance(str2,float)  or isinstance(str3,float):
        return np.NaN
    return len((str1&str2)-str3)/len(str1|str2|str3)
    
def union(strs):
    str1,str2,str3 = strs
    if isinstance(str1,float) or isinstance(str2,float)  or isinstance(str3,float):
        return np.NaN

    return (len(str3&(str1-str2))+len(str3&(str2-str1)))/len(str1|str2|str3)
    
    

def create_similarity(data,jaccard=jaccard,create_set=True,stop_words={},tokenize=True):
    """
    Calculate Jaccard index with previous stories
    """
    #Add original story
    data["story_original"] = story_original
    
    if tokenize:
        for col in ["story1","story2","story3","story_merged","story_original"]:
            data.loc[:,col] = data.loc[:,col].apply(story2set,create_set=create_set,stop_words=stop_words)
    
    
    #Similarity with story1
    data["jaccard_1"] = data[["story1","story_merged"]].apply(jaccard,axis=1)
    #Similarity with story2
    data["jaccard_2"] = data[["story2","story_merged"]].apply(jaccard,axis=1)
    #Similarity with story2
    data["jaccard_3"] = data[["story3","story_merged"]].apply(jaccard,axis=1)
    #Average similarity with previous layer
    data["jaccard_previous_layer"] = data[["jaccard_1","jaccard_2","jaccard_3"]].max(1)
    #Similarity with story_original
    data["jaccard_original_story"] = data[["story_merged","story_original"]].apply(jaccard,axis=1)

    #With original story
    data["jaccard_os_1"] = data[["story1","story_original"]].apply(jaccard,axis=1)
    data["jaccard_os_2"] = data[["story2","story_original"]].apply(jaccard,axis=1)
    data["jaccard_os_3"] = data[["story3","story_original"]].apply(jaccard,axis=1)
    

    #Similariry within layer
    data["jaccard_wl_12"] = data[["story1","story2"]].apply(jaccard,axis=1)
    data["jaccard_wl_13"] = data[["story1","story3"]].apply(jaccard,axis=1)
    data["jaccard_wl_23"] = data[["story2","story3"]].apply(jaccard,axis=1)
    data["jaccard_within_layer"] = data[["jaccard_wl_12","jaccard_wl_13","jaccard_wl_23"]].mean(1)

    return data


def create_interactions(data):
    """
    Find motifs 
    """
    
    data["propagation"] =data[["story1","story2","story_merged"]].apply(propagation,axis=1)
    data["inference"] =data[["story1","story2","story_merged"]].apply(inference,axis=1)
    data["forgetting"] =data[["story1","story2","story_merged"]].apply(forgetting,axis=1)
    data["union"] =data[["story1","story2","story_merged"]].apply(union,axis=1)
    return data
    
    
    
def create_similarity_indep_replicates(results, sim_function=jaccard):

    av_jac = lambda x: [sim_function(stories) for stories in list(itertools.combinations(x,2))]
    results = results.sort_values(by=["condition","rep"], ascending=False)
    results["k"] = results.index%3
    results.loc[results["condition"].str.contains("Chain"),"k"] = 0

    r = results.groupby(["condition","layer_n","k"])["story_merged"].agg(av_jac).reset_index()
    r = r.explode("story_merged").sort_values(by="condition")
    # # r = r.groupby(["condition","layer_n"]).mean().reset_index()
    r["story_merged"] = r["story_merged"].astype(float)
    
    r = r.reset_index(drop=True)
    return r
    
def plot_similarity_indep_replicates(r, ax=None):
    #g = sns.pointplot(x="layer_n",y="story_merged",hue="condition",data=r,lw=2,s=10,
                 #hue_order=conditions,palette=cond2color,ax=ax)
    #plt.setp(g.lines, alpha=0.3)

    sns.lineplot(x="layer_n",y="story_merged",hue="condition",
             palette=cond2color, data=r,lw=2,marker="o",markersize=7)
    
    set_style("Similarity between independent stories")
    
def main_similarity_indep_replicates(results, sim_function=jaccard, ax=None):
    r = create_similarity_indep_replicates(results, sim_function=sim_function)
    
    plot_similarity_indep_replicates(r, ax=ax)
    return r
    
    

