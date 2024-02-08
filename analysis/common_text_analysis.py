from common_variables import *

import os
from sentence_transformers import SentenceTransformer, util
import string
from collections import Counter
import itertools
from sklearn.metrics.pairwise import cosine_similarity


import pickle
global_emb = pickle.load(open(f"{path_text_embeddings}/story_embeddings_global_emb.pickle", "rb"))

# import these modules
from nltk.stem import PorterStemmer
ps = PorterStemmer()

## CREATE SIMILARITY

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
    return len(str1 & str2) / len(str1 | str2)

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
    
def transformer_similarity(sentence1, sentence2, model):
    if sentence1 in global_emb:
        embeddings1 = global_emb[sentence1]
    else:
        embeddings1 = model.encode([sentence1], convert_to_tensor=True)
        global_emb[sentence1] = embeddings1
    
    if sentence2 in global_emb:
        embeddings2 = global_emb[sentence2]
    else:
        embeddings2 = model.encode([sentence2], convert_to_tensor=True)
        global_emb[sentence2] = embeddings2
    
    #Compute cosine-similarits
    return float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])
    

#Transformer model
def w2t(strs,model=model_t):
    str1,str2 = strs
    if isinstance(str1,float) or isinstance(str2,float):
        return np.NaN
    return transformer_similarity(str1,str2, model)

def create_similarity(data,fun_similarity=jaccard,create_set=True,stop_words={},tokenize=True):
    """
    Calculate similarity with previous stories
    """    
    #Add original story
    data["story_original"] = story_original
    
    if tokenize:
        for col in ["story1","story2","story3","story_merged","story_original"]:
            data.loc[:,col] = data.loc[:,col].apply(story2set,create_set=create_set,stop_words=stop_words)
    
    
    #Similarity with story1
    data["sim_1"] = data[["story1","story_merged"]].apply(fun_similarity,axis=1)
    #Similarity with story2
    data["sim_2"] = data[["story2","story_merged"]].apply(fun_similarity,axis=1)
    #Similarity with story2
    data["sim_3"] = data[["story3","story_merged"]].apply(fun_similarity,axis=1)
    #Average similarity with previous layer
    data["sim_previous_layer"] = data[["sim_1","sim_2","sim_3"]].max(1)
    #Similarity with story_original
    data["sim_original_story"] = data[["story_merged","story_original"]].apply(fun_similarity,axis=1)

    #With original story
    data["sim_os_1"] = data[["story1","story_original"]].apply(fun_similarity,axis=1)
    data["sim_os_2"] = data[["story2","story_original"]].apply(fun_similarity,axis=1)
    data["sim_os_3"] = data[["story3","story_original"]].apply(fun_similarity,axis=1)
    

    #Similariry within layer
    data["sim_wl_12"] = data[["story1","story2"]].apply(fun_similarity,axis=1)
    data["sim_wl_13"] = data[["story1","story3"]].apply(fun_similarity,axis=1)
    data["sim_wl_23"] = data[["story2","story3"]].apply(fun_similarity,axis=1)
    data["sim_within_layer"] = data[["sim_wl_12","sim_wl_13","sim_wl_23"]].mean(1)

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
    
    
    
def create_similarity_indep_replicates(results, fun_similarity=jaccard):

    av_sim = lambda x: [fun_similarity(stories) for stories in list(itertools.combinations(x,2))]
    results = results.sort_values(by=["condition","rep"], ascending=False)
    results["k"] = results.index%3
    results.loc[results["condition"].str.contains("Chain"),"k"] = 0

    r = results.groupby(["condition","layer_n","k"])["story_merged"].agg(av_sim).reset_index()
    r = r.explode("story_merged").sort_values(by="condition")
    # # r = r.groupby(["condition","layer_n"]).mean().reset_index()
    r["story_merged"] = r["story_merged"].astype(float)
    
    r = r.reset_index(drop=True)
    return r
    

    
def main_similarity_indep_replicates(results, fun_similarity=jaccard, ax=None):
    r = create_similarity_indep_replicates(results, fun_similarity=fun_similarity)
    
    plot_similarity_indep_replicates(r, ax=ax)
    return r
    
    


