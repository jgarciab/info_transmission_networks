from common_variables import *
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import pylab as plt
import scipy.stats as st
from scipy.stats import binom 

########################## DETERMINISTIC MODEL 
def deterministic_model(p, p_after, max_generations=6):
    """
    Calculate the decay of probability values in a network.
    
    Parameters:
    - p (list of floats): Probabilities of word transmission reading the story 0, 1, 2, or 3.
    - max_generations (int, optional): Maximum number of generations. Default is 6.
    
    Returns:
    float: A decayed probability value.
    """
    
    # Calculate probability mass function for the last element of p
    freq = binom.pmf(range(4), 3, p)
    # print(freq)
    # Create a matrix of probability mass functions
    matrix = np.vstack([binom.pmf(range(4), 3, prob) for prob in p_after]).T

    # Handle edge cases
    if p == 1:
        return 1#np.inf
    if p == 0:
        return 0
    
    
    for i in range(max_generations-1):
        freq = matrix @ freq
    
    
    return np.array(p_after) @ freq


def deterministic_model_plot(k, xmin=0.5, generations=1):
    colors = sns.color_palette("Greens",k)
    
    #Probability of remembering
    p_values = np.linspace(xmin,1,1000)

    #Plot binomial
    for i in range(k):
        if i == 0:
            label=f"Intersection"
        elif i == (k-1):
            label=f"Union"
        else:
            label=f"At least {k-i}"
        
        c = colors.pop(0)
    
        #Binomial uses probability of forgetting (0 = all remember, 1 = at least one remember)
        plt.plot(p_values,binom.cdf(i,k,1-p_values)**generations,"-",color=c,label=label,lw=2)
    
        

    #1:1 line
    #plt.plot([xmin,1],[1,1],"--",color="gray",zorder=0)
    plt.plot(p_values,p_values**generations,"--",color="gold",label=None,zorder=0)
    plt.plot(p_values,p_values,"--",color="gray",label=None,zorder=0)

    #plt.yscale("log")
    
    #Legend
    plt.legend(loc=1, ncol=1, title="Remembered if:", fontsize=12, title_fontsize=12)
    #plt.legend(loc='center left', bbox_to_anchor=(0.65, 0.2), ncol=1, title="Remembered if:", fontsize=12, title_fontsize=12)
    #Labels
    plt.xlabel("Baseline probability")
    plt.ylabel(f"Frequency at generation {generations+1}")
    sns.despine(bottom=True,left=True)

######################################## CATEGORICAL MODEL

# DEFINE FITTED MODEL PARAMETERS
from dataclasses import dataclass
@dataclass
class ModelData:
    beta_n_observed: [float, float, float] #increase with fraction of previous stories with the word
    beta_network: float #fixed effect on network condition
    beta_n_words: float #decay on number of words read
    story_length: int 

# This comes from the R code
model_data = ModelData(
    beta_n_observed=[-2.32,-1.45,-0.90],
    beta_network=0.66,
    beta_n_words=-0.18,
    story_length=265
)

def create_prob(prob, beta_n_observed=model_data.beta_n_observed, beta_network=model_data.beta_network, 
    n_remembered=1, network=0, n_words=265, beta_n_words=model_data.beta_n_words,
    chain_less_words_factor=3,story_length=model_data.story_length):
    """
    Adjusts the given probability value using a statistical model, assuming all variables stay constant except the parameters below. 
    Takes as baseline the baseline probability.
    
    Parameters: 
    - prob (float): p_{j,1} - Input probability value to be adjusted. Should be between 0 and 1.
    - beta_network (float, optional): beta_C - parameter for reading the story several times.
    - beta_n_observed (float, optional): beta_N - parameter for reading the story several times. 
    - beta_n_words (float, optional): beta_S - parameter for number of words read. 
    - network (int, optional): C - Indicates whether the data is in a network (1) or not (0). Default is 0.
    - n_words (int, optional): n_w - Number of words in the original text. Default is 265.
    - n_remembered (float, optional): N - number of times the event is remembered. Default is 1.
    - story_length (int, optional): S: current number of words in story
    
    Returns:
    - float: Adjusted probability value.
    """
    
#     MAKE SURE NUMBER OF WORDS IS AT LEAST 1 (CANNOT PASS 0 WORDS)
    n_words=np.where(n_words>0, n_words, 1)
    
#     TESTING if n_remembered is a list (chain vs network conditions)
    if isinstance(n_remembered, int):
        n_remembered_vec=int(n_remembered>0) # Indicates whether at least 1 person remembered
        beta_n_observed_vec=beta_n_observed[n_remembered-1] # selects parameter value corresponding to N
    else:
        n_remembered_vec=np.array([int(k>0) for k in n_remembered])
        beta_n_observed_vec=np.array([beta_n_observed[k-1] for k in n_remembered])
        
#     DEFINE PRIOR ADJUSTED BY RANDOM INTERCEPTS
    beta_0= np.log(prob / (1 - prob)) - beta_n_observed[2] - beta_n_words * np.log(story_length)
    # Apply corrections to the log odds using the statistical model   
    odds = np.exp(
     beta_0            # additive constant from the prior 
     + n_remembered_vec * beta_n_observed_vec           # Adjust based on how many times an event is remembered
     + network * beta_n_words * np.log(chain_less_words_factor) +  # Adjust for being in a network, assuming 3x more words are read
     + network * beta_network     # Adjust using network condition indicator
     + beta_n_words * np.log(n_words)     # Adjust based on the number of words
    )
   
    # Convert the corrected odds back to a probability
    adjusted_prob = odds / (odds + 1)

    return adjusted_prob

def create_probs(prob, beta_n_observed=model_data.beta_n_observed, n_words=265, network=1, fear=1, story_length=265):
    """
    Create transmission probabilities if the story has been read 1, 2, or 3 times. (from the story being read 1 time in the chain)
    
    Parameters:
    - prob (float): Input probability.
    - decay_value (float, optional): Decay value. Default is 1.
    
    Returns:
    list of floats: A list of probabilities.
    """
    
    n_remembered = np.linspace(0,1,4)[1:]
    
    if prob == 1:
        return [0, 1, 1, 1]

    r = create_prob(prob, beta_n_observed=beta_n_observed, n_words=n_words, network=1, n_remembered=n_remembered,story_length=story_length)
    

    # return probabilities
    return [0] + list(r)


def get_base_freqs():
    """
    Extract distribution of baseline transmission probabilities from the data.
    
    Parameters:
    
    Returns:
    list of probabilities.
    """
    
    path_baseline_probs="../data/word_baseline_probs.csv"
    base_freq = pd.read_csv(path_baseline_probs, sep="\t")
    base_freq = base_freq[["word","baseline_prob_3_word_265"]]
    base_freqs = base_freq["baseline_prob_3_word_265"]
    return base_freqs

def bdd_exponential(story_length_=100, decay=1, minval=0,maxval=1):
    """
    Returns a list of bounded exponential variable.
    
    Parameters:
    - story_length_ (int): number samples to produce
    - decay (float): decay rate of exponential RV
    - minval (int): minimum of exponential distribution
    - maxval (int): maximum of exponential distribution
    
    Returns:
    list of probabilities.
    """
    # This is a map from a uniform random variable to a bounded exponential between mimnval and maxval.
#     https://stats.stackexchange.com/questions/508749/generating-random-samples-obeying-the-exponential-distribution-with-a-given-min
    u = np.random.uniform(0, 1, story_length_)
    x = -(1/decay)*np.log(np.exp(-decay*minval) + u * (np.exp(-decay*maxval) - np.exp(-decay*minval)))
    return x

def unif_random(story_length_=100):
    """
    Returns a list of uniform random numbers.
    
    Parameters:
    - story_length_ (int): number samples to produce
    
    Returns:
    list of probabilities.
    """
    prob = np.random.random(story_length)
    return prob


def fitted_exponential(story_length_,minval=0,maxval=0.9):
    """
    Generates a list of bounded exponential variable with decay rate fitted from the data.
    
    Parameters:
    - story_length_ (int): number samples to produce
    - minval (int): minimum of exponential distribution
    - maxval (int): maximum of exponential distribution
    
    Returns:
    list of probabilities.
    """
    base_freqs=get_base_freqs()
    loc, scale = sp.stats.expon.fit(base_freqs, floc=0)
    prob=bdd_exponential(story_length_=story_length_,decay=1/scale,minval=minval,maxval=maxval)
    return prob

def jaccard_similarity(a,b):
    """
    Finds Jaccard similarity between two binary strings
    
    Parameters:
    - a (float list): first binary string
    - b (float list): second binary string
    
    Returns:
    similarity measure in [0,1] (float).
    """
    a= a==1
    b= b==1
    if (np.sum(a|b)==0):
        jaccard_sim=0
    else:            
        jaccard_sim=np.sum(a&b)/np.sum(a|b)
    return jaccard_sim

def calculate_similarity(all_results, original=False, fun="mean",story_length=265):
    """
    Finds Jaccard similarity values in simulated data-set
    
    Parameters:
    - all_results (float list): list of simulated data
    - original (bool): True computes seed-similarity, False computes replicate similarity
    - fun (function): "mean" computes mean of similarity values, "std" computes 
    - story_length (int): length of original seed text 
    
    Returns:
    similarity mean or std for each generation (float list).
    """
    layers = all_results[0].shape[1]
    n_reps = len(all_results)
    #Similarity
    sim = []
    for l in range(layers):
        v_l = []
        
        if original:
            for i in range(n_reps):
                if len(all_results[0].shape) == 3:
                    random_agent=np.random.randint(0,3)
                    r1 = all_results[i][:, :, random_agent]
                else:
                    r1 = all_results[i][:, :]
#                 r2  = np.ones(len(r1[:,0])).astype(bool)
#                 r2[story_length:] = False
#                 v_l.append(jaccard_similarity(r1[:, l], r2))

                r2  = np.ones(story_length).astype(bool)
                v_l.append(jaccard_similarity(r1[:story_length, l], r2))
        else:
            for i in range(n_reps):
                for j in range(i+1, n_reps):
                    if len(all_results[0].shape) == 3:
                        random_agent_1=np.random.randint(0,3)
                        random_agent_2=np.random.randint(0,3)
                        r1 = all_results[i][:, :, random_agent_1]
                        r2 = all_results[j][:, :, random_agent_2]
                    else:
                        r1 = all_results[i][:, :]
                        r2 = all_results[j][:, :]
                    
                    v_l.append(jaccard_similarity(r1[:, l], r2[:, l]))
        if fun == "mean":
            sim.append(np.mean(v_l))
        elif fun == "std":
            sim.append(np.std(v_l))
        else:
            print("Please enter mean or std as fun input")
    return sim


def calculate_sim_ci(all_results, original=False, story_length=265):
    """
    Finds confidence intervals (CI) for Jaccard similarity values
    
    Parameters:
    - all_results (float list): list of simulated data
    - original (bool): True computes CI of seed-similarity, False computes CI of replicate similarity
    - story_length (int): length of original seed text 
    
    Returns:
    similarity confidence interval [lower CI,upper CI] (float list).
    """
    layers = all_results[0].shape[1]
    n_reps = len(all_results)
    #Similarity
    sim_ci_lower = []
    sim_ci_upper = []
    for l in range(layers):
        v_l = []
        
        if original:
            for i in range(n_reps):
                if len(all_results[0].shape) == 3:                    
                    random_agent=np.random.randint(0,3)
                    r1 = all_results[i][:, :, random_agent]
                else:
                    r1 = all_results[i][:, :]
#                 r2  = np.ones(len(r1[:,0])).astype(bool)
#                 r2[story_length:] = False
#                 v_l.append(jaccard_similarity(r1[:, l], r2))

                r2  = np.ones(story_length).astype(bool)
                v_l.append(jaccard_similarity(r1[:story_length, l], r2))
        else:
            for i in range(n_reps):
                for j in range(i+1, n_reps):
                    if len(all_results[0].shape) == 3:                    
                        random_agent_1=np.random.randint(0,3)
                        random_agent_2=np.random.randint(0,3)
                        r1 = all_results[i][:, :, random_agent_1]
                        r2 = all_results[j][:, :, random_agent_2]
                    else:
                        r1 = all_results[i][:, :]
                        r2 = all_results[j][:, :]
                    
                    v_l.append(jaccard_similarity(r1[:, l], r2[:, l]))
        sim_ci=st.t.interval(alpha=0.95, df=len(v_l)-1, loc=np.mean(v_l), scale=st.sem(v_l))
        sim_ci_lower.append(sim_ci[0])
        sim_ci_upper.append(sim_ci[1])

    return sim_ci_lower,sim_ci_upper

def simulate_chain(prob, beta_n_observed=model_data.beta_n_observed, layers=6, story_length=1000, seed_length=265, introduction_rate=0.05):
    """
    Simulates chain condition of experiment.
    
    Parameters:
    - prob (float list): Input baseline probability values - i.e. the prior. Values should be between 0 and 1.
    - beta_n_observed (float, optional): beta_N - parameter for reading the story several times. 
    - layers (int): Number of generations in the experiment
    - story_length (int): maximum possible number of words in text (adjusting seed length according to extra word dictionary)
    - seed_length (int): length of original seed text
    - introduction_rate (float): probability for new words to be introduced at each generation.
    
    Returns:
    story (bool list): matrix of words present at each generation (1 - present, 0 - absent).
    """
    #Chain
    story = np.ones((story_length, layers)).astype(bool)

    # First layer - transmission threshold defined by baseline transmission probabilities
    story[:, 0] = (np.random.random(story_length)<prob)

    # Set newly added elements to false
    story[seed_length:, 0] = False 
    
    # Check whether any new words are introduced
    introductions = (np.random.random(story_length)<introduction_rate)
    # Set original story elements to false
    introductions[:seed_length] = False 
    # Combine original and newly added words
    story[:, 0] |= introductions

    for i in range(1, layers):
        # Compute transmission probability - NOTE n_remembered=3 to match to network
        p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=story[:, i-1].sum(), 
                        network=0, n_remembered=3,story_length=seed_length) 
        # Check whether any new words are introduced
        introductions = (np.random.random(story_length)<introduction_rate)
        # Set original story elements to false
        introductions[:seed_length] = False
        
        # Update stories with transmitted words
        story[:, i] = (story[:, i-1] & (np.random.random(story_length)<p)) | introductions
    return story

        
def simulate_net(prob, beta_n_observed=model_data.beta_n_observed, layers=6, 
                 story_length=1000, seed_length=265, introduction_rate=0.05):
    """
    Simulates network condition of experiment.
    
    Parameters:
    - prob (float list): Input baseline probability values - i.e. the prior. Values should be between 0 and 1.
    - beta_n_observed (float, optional): beta_N - parameter for reading the story several times. 
    - layers (int): Number of generations in the experiment
    - story_length (int): maximum possible number of words in text (adjusting seed length according to extra word dictionary)
    - seed_length (int): length of original seed text
    - introduction_rate (float): probability for new words to be introduced at each generation.
    
    Returns:
    story_n (bool list): matrix of words present at each generation (1 - present, 0 - absent).
    """
    #Network
    story_n = np.ones((story_length, layers, 3)).astype(bool)
    
    # First layer - transmission threshold defined by baseline transmission probabilities
    story_n[:, 0, 0] = (np.random.random(story_length)<prob) 
    story_n[:, 0, 1] = (np.random.random(story_length)<prob) 
    story_n[:, 0, 2] = (np.random.random(story_length)<prob) 
    # Set newly added elements to false
    story_n[seed_length:,0,:] = False

    # Check whether any new words are introduced for each participant in a generation
    for r in range(3):
        introductions = (np.random.random(story_length)<introduction_rate)
        introductions[:seed_length] = False
        # Combine original and newly added words
        story_n[:, 0, r] |= introductions

    for i in range(1, layers):
        # Check how many participants remembered words in the previous generation
        keep1 = story_n[:, i-1, 0]
        keep2 = story_n[:, i-1, 1]
        keep3 = story_n[:, i-1, 2]
        sum_keep = story_n[:, i-1, :].sum(1)

        # Loop over participants in the current generation
        for j in range(3):
            # Compute transmission probabilities 
            # Note n_words divided by 3 because correct_odds assumes the newtork reads 3 times more words
            p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=(sum_keep).sum()/3, 
                            network=1, n_remembered=sum_keep) 
            
            # Determine whether words are kept according to union of individual participants in previous generation
            keep = keep1|keep2|keep3 
            keep &= np.random.random(story_length)<p
            # Check for newly introduced words
            introductions = (np.random.random(story_length)<introduction_rate)
            introductions[:seed_length] = False
            keep |= introductions

            # Update stories with transmitted words
            story_n[:, i, j] = keep
        
    return story_n


def simulate_experiment(prob, layers = 6, seed_length=265, n_reps = 100, extra_w = 20, extra_intro_rate = 0.05, 
                        extra_baseline_prob = 0.1, netcolor='#3d348b', chaincolor='#e6af2e'):
    """
    Simulates Single-pathway (chain) and Multiple-pathway (network) conditions of experiment.
    
    Parameters:
    - prob (float list): list of baseline probability values for words in the original text
    - layers (int): Number of generations in the experiment
    - seed_length (int): length of original seed text
    - n_reps (int): number of replicates for each experimental condition
    - extra_w (int): number of extra words that can be added to the story from a set dictionary.
    - extra_intro_rate (float): probability for new words to be introduced at each generation.
    - extra_baseline_prob (float): delta - baseline probability for all newly added words.
    - netcolor (color string): hexcode for colour of plotted Multiple-pathway condition data
    - chaincolor (color string): hexcode for colour of plotted Single-pathway condition data
    
    Returns:
    null
    """
    
    ## RESHAPE BASELINE PROBABILITY LIST TO ADD IN POSSIBLE EXTRA WORDS
    prob = np.concatenate([prob, extra_baseline_prob+np.zeros(extra_w)])
    total_length = seed_length + extra_w # NEW LENGTH OF INFORMATION VECTOR

    #     PLOT DISTRIBUTION OF BASELINE PROBABILITIES
    sns.histplot(prob, bins=np.linspace(0,1,20), color="cornflowerblue", stat="percent")
    plt.yscale("log")
    
    # Define plots
    fig1 = plt.figure(figsize=(16,3))
    fig2 = plt.figure(figsize=(16,3))
    fig3 = plt.figure(figsize=(16,3))
    fig4 = plt.figure(figsize=(16,3))
    
    ylabel=True

    # Initialise results matrices
    all_results_n = []
    all_results_c = []
    results_n = np.zeros((n_reps,total_length))
    results_c = np.zeros((n_reps,total_length))
    words_n = np.zeros((n_reps,layers))
    words_c = np.zeros((n_reps,layers))

    #     Simulate Single-pathway and Multiple-pathway conditions for each replicate
    for i in range(n_reps):      
        sc = simulate_chain(prob, beta_n_observed=model_data.beta_n_observed, layers=layers, 
                       story_length=total_length, seed_length=seed_length, introduction_rate=extra_intro_rate)
        sn = simulate_net(prob, beta_n_observed=model_data.beta_n_observed, layers=layers, 
                       story_length=total_length, seed_length=seed_length, introduction_rate=extra_intro_rate)
        all_results_n.append(sn)
        all_results_c.append(sc)

    #     compute averages for Single-pathway and Multiple-pathway conditions in each replicate
    for i in range(n_reps): 
        results_n[i,:] = all_results_n[i][:, -1, :].mean(1)
        results_c[i,:] = all_results_c[i][:, -1]
        words_n[i,:] = all_results_n[i].mean(2).sum(0)
        words_c[i,:] = all_results_c[i].sum(0)

    # average across replicates
    results_n = results_n.mean(0)
    results_c = results_c.mean(0)
    
#     Plot word frequency for each word in each condition
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(prob,results_n,".",label="Multiple-pathway",color=netcolor)
    ax1.plot(prob,results_c,".",label="Single-pathway",color=chaincolor)
    ax1.legend()
    ax1.set_xlabel("$p_{i,1}$, Baseline probability")
    if ylabel:
        ax1.set_ylabel(f"Word frequency in layer {layers}")
    sns.despine(ax=ax1)
    ax1.plot([0,1],[0,1],"--",color="lightgray")

#     Plot average number of words for each condition
    ax2 = fig2.add_subplot(1,3,1)
    ax2.plot(range(layers),words_n.mean(0),label="Multiple-pathway",color=netcolor)
    ax2.plot(range(layers),words_c.mean(0),label="Single-pathway",color=chaincolor)
    ax2.legend()
    ax2.set_xlabel("$k$, Generation")
    if ylabel:
        ax2.set_ylabel("Number of words")
    sns.despine(ax=ax2)

    #     Compute replicate similarities along with confidence intervals for each condition 
    sim_n = calculate_similarity(all_results_n) 
    sim_c = calculate_similarity(all_results_c) 
    sim_n_ci_lower,sim_n_ci_upper=calculate_sim_ci(all_results_n)
    sim_c_ci_lower,sim_c_ci_upper=calculate_sim_ci(all_results_c)
    yerr_n=np.empty((2,layers))
    yerr_n[0,:]=sim_n-np.array(sim_n_ci_lower)
    yerr_n[1,:]=np.array(sim_n_ci_upper)-sim_n
    yerr_c=np.empty((2,layers))
    yerr_c[0,:]=sim_c-np.array(sim_c_ci_lower)
    yerr_c[1,:]=np.array(sim_c_ci_upper)-sim_c
    
#     Plot replicate similarities with confidence intervals
    ax3 = fig3.add_subplot(1,3,1)
    ax3.plot(range(1, layers+1),sim_n,label="Multiple-pathway",marker='o',color=netcolor)
    ax3.fill_between(range(1, layers+1), y1=sim_n_ci_lower,y2=sim_n_ci_upper, alpha=0.3)
    ax3.plot(range(1, layers+1),sim_c,label="Single-pathway",marker='o',color=chaincolor)
    ax3.fill_between(range(1, layers+1), y1=sim_c_ci_lower,y2=sim_c_ci_upper, alpha=0.3)
    ax3.legend()
    ax3.set_xlabel("$k$, Generation")
    if ylabel:
        ax3.set_ylabel("$\\theta_r$, Replicate-similarity")
    sns.despine(ax=ax3, bottom=True, left=True)
    ax3.grid(axis="y")

    #     Compute seed similarities along with confidence intervals for each condition 
    sim_n = calculate_similarity(all_results_n, original=True,story_length=seed_length) 
    sim_c = calculate_similarity(all_results_c, original=True,story_length=seed_length) 
    sim_n_ci_lower,sim_n_ci_upper=calculate_sim_ci(all_results_n, original=True,story_length=seed_length)
    sim_c_ci_lower,sim_c_ci_upper=calculate_sim_ci(all_results_c, original=True,story_length=seed_length)
    yerr_=np.empty((2,layers))
    yerr_n[0,:]=sim_n-np.array(sim_n_ci_lower)
    yerr_c[1,:]=np.array(sim_n_ci_upper)-sim_n

    #     Plot seed similarities with confidence intervals
    ax4 = fig4.add_subplot(1,3,1)
    ax4.plot(range(1, layers+1),sim_n,label="Multiple-pathway",marker='o',color=netcolor)
    ax4.fill_between(range(1, layers+1), y1=sim_n_ci_lower,y2=sim_n_ci_upper, alpha=0.3)
    ax4.plot(range(1, layers+1),sim_c,label="Single-pathway",marker='o',color=chaincolor)
    ax4.fill_between(range(1, layers+1), y1=sim_c_ci_lower,y2=sim_c_ci_upper, alpha=0.3)
    ax4.legend()
    ax4.set_xlabel("$k$, Generation")
    if ylabel:
        ax4.set_ylabel("$\\theta_s$, Seed-similarity")
    sns.despine(ax=ax4, bottom=True, left=True)
    ax4.grid(axis="y")

# OPTIONAL - SAVE FIGURES TO results FOLDER
    figures_path="results/"
#     fig1.savefig(f"{figures_path}ABM_initialWordDistribution.pdf", bbox_inches="tight")   
#     fig3.savefig(f"{figures_path}ABM_.pdf", bbox_inches="tight")   
    fig3.savefig(f"{figures_path}ABM_replicatesimilarity.pdf", bbox_inches="tight")   
    fig4.savefig(f"{figures_path}ABM_seedsimilarity.pdf", bbox_inches="tight")  









######################################## CONTINUOUS MODEL

# def create_prob(prob, beta_n_observed=model_data.beta_n_observed, beta_network=model_data.beta_network,
#     n_remembered=1, network=0, n_words=265, beta_n_words=model_data.beta_n_words,
#     chain_less_words_factor=3,story_length=model_data.story_length):
#     """
#     Adjusts the given probability value using a statistical model, assuming all variables stay constant except the parameters below. 
#     Takes as baseline the baseline probability.
    
#     Parameters:
#     - prob (float): p_{j,1} - Input probability value to be adjusted. Should be between 0 and 1.
#     - beta_network (float, optional): beta_1 - parameter for reading the story several times.
#     - beta_n_observed (float, optional): beta_2 - parameter for reading the story several times. Default is 1.337.
#     - beta_n_words (float, optional): beta_3 - parameter for number of words read. Default is 1.5681.
#     - network (int, optional): C - Indicates whether the data is in a network (1) or not (0). Default is 0.
#     - n_words (int, optional): n_w - Number of words in the original text. Default is 265.
#     - n_remembered (float, optional): N - Fraction of times the event is remembered. Should be between 0 and 1. Default is 1.
#     - story_length (int, optional): sum_j w_{j,k}: current number of words in story
    
#     Returns:
#     - float: Adjusted probability value.
#     """
#     # np.log(prob / (1 - prob)) = (
#     #     alpha_0 +
#     #     beta_n_observed * 1 +                   # Adjust based on how manytimes an event is remembered
#     #     beta_n_words * np.log(story_length)     # Adjust based on the number of words
#     # )

#     # Get alpha0 from (equation above)
#     alpha_0 = np.log(prob / (1 - prob)) - beta_n_observed - beta_n_words * np.log(story_length)
    
#     # Apply corrections to the log odds using the statistical model
#     odds = np.exp(
#      alpha_0 
#      + beta_n_observed * n_remembered           # Adjust based on how many times an event is remembered
#      + network * beta_n_words * np.log(chain_less_words_factor) +  # Adjust for being in a network, assuming 3x more words are read
#      + network * beta_network
#      + beta_n_words * np.log(n_words)     # Adjust based on the numberof words
#     )
    
#     # Convert the corrected odds back to a probability
#     adjusted_prob = odds / (odds + 1)

#     return adjusted_prob

# def create_probs(prob, beta_n_observed=model_data.beta_n_observed, n_words=265, network=1, story_length=265):
#     """
#     Create transmission probabilities if the story has been read 1, 2, or 3 times. (from the story being read 1 time in the chain)
    
#     Parameters:
#     - prob (float): Input probability.
#     - decay_value (float, optional): Decay value. Default is 1.
    
#     Returns:
#     list of floats: A list of probabilities.
#     """
    
#     n_remembered = np.linspace(0,1,4)[1:]
    
#     if prob == 1:
#         return [0, 1, 1, 1]

#     r = create_prob(prob, beta_n_observed=beta_n_observed, n_words=n_words, network=1, n_remembered=n_remembered,story_length=story_length)
    

#     # return probabilities
#     return [0] + list(r)


# def get_base_freqs():
#     """
#     Extract distribution of baseline transmission probabilities from the data.
    
#     Parameters:
    
#     Returns:
#     list of probabilities.
#     """
#     path_baseline_probs="survival_analysis/word_baseline_probs.csv"
#     base_freq = pd.read_csv(path_baseline_probs, sep="\t")
#     base_freq = base_freq[["word","baseline_prob_3_word_265"]]
#     base_freqs = base_freq["baseline_prob_3_word_265"]
#     return base_freqs

# def bdd_exponential(story_length_=100, decay=1, minval=0,maxval=1):
#     """
#     Returns a list of bounded exponential variable.
    
#     Parameters:
#     - story_length_ (int): number samples to produce
#     - decay (float): decay rate of exponential RV
#     - minval (int): minimum of exponential distribution
#     - maxval (int): maximum of exponential distribution
    
#     Returns:
#     list of probabilities.
#     """
#     # This is a map from a uniform random variable to a bounded exponential between mimnval and maxval.
# #     https://stats.stackexchange.com/questions/508749/generating-random-samples-obeying-the-exponential-distribution-with-a-given-min
#     u = np.random.uniform(0, 1, story_length_)
#     x = -(1/decay)*np.log(np.exp(-decay*minval) + u * (np.exp(-decay*maxval) - np.exp(-decay*minval)))
#     return x

# def unif_random(story_length_=100):
#     prob = np.random.random(story_length)
#     return prob


# def fitted_exponential(story_length_,minval=0,maxval=0.9):
#     """
#     Generates a list of bounded exponential variable with decay rate fitted from the data.
    
#     Parameters:
#     - story_length_ (int): number samples to produce
#     - decay (float): decay rate of exponential RV
#     - minval (int): minimum of exponential distribution
#     - maxval (int): maximum of exponential distribution
    
#     Returns:
#     list of probabilities.
#     """
#     base_freqs=get_base_freqs()
#     loc, scale = sp.stats.expon.fit(base_freqs, floc=0)
#     prob=bdd_exponential(story_length_=story_length_,decay=1/scale,minval=minval,maxval=maxval)
#     return prob


# def fitted_lognormal(story_length_=100,maxval=0.9):
#     base_freqs=get_base_freqs()
#     shape, loc, scale = sp.stats.lognorm.fit(base_freqs, loc=0)
#     lognmu, lognsigma=np.log(scale), shape
#     prob=np.random.lognormal(mean=lognmu,sigma=lognsigma,size=10*story_length_)
#     prob=prob[prob<maxval] #Truncate values below a threshold (note in general they can be sampled as prob>1)
#     prob = np.random.choice(prob, size=story_length_, replace=False)
#     return prob




# def jaccard_similarity(a,b):
#     """
#     Finds Jaccard similarity between two binary strings
    
#     Parameters:
#     - a (float list): first binary string
#     - b (float list): second binary string
    
#     Returns:
#     similarity measure in [0,1] (float).
#     """
#     a= a==1
#     b= b==1
#     if (np.sum(a|b)==0):
#         jaccard_sim=0
#     else:
#         jaccard_sim=np.sum(a&b)/np.sum(a|b)
#         # jaccard_sim=2*np.sum((a&b))/265
#     return jaccard_sim

# def calculate_similarity(all_results, original=False, fun="mean",story_length=265):
#     layers = all_results[0].shape[1]
#     n_reps = len(all_results)
#     #Similarity
#     sim = []
#     for l in range(layers):
#         v_l = []
        
#         if original:
#             for i in range(n_reps):
#                 if len(all_results[0].shape) == 3:
#                     random_agent=np.random.randint(0,3)
#                     r1 = all_results[i][:, :, random_agent]
#                 else:
#                     r1 = all_results[i][:, :]
#                 r2  = np.ones(len(r1[:,0])).astype(bool)
#                 r2[story_length:] = False
#                 v_l.append(jaccard_similarity(r1[:, l], r2))
#         else:
#             for i in range(n_reps):
#                 for j in range(i+1, n_reps):
#                     if len(all_results[0].shape) == 3:
#                         random_agent_1=np.random.randint(0,3)
#                         random_agent_2=np.random.randint(0,3)
#                         r1 = all_results[i][:, :, random_agent_1]
#                         r2 = all_results[j][:, :, random_agent_2]
#                     else:
#                         r1 = all_results[i][:, :]
#                         r2 = all_results[j][:, :]
                    
#                     v_l.append(jaccard_similarity(r1[:, l], r2[:, l]))
#         if fun == "mean":
#             sim.append(np.mean(v_l))
#         elif fun == "std":
#             sim.append(np.std(v_l))
#     return sim


# def calculate_sim_ci(all_results, original=False, story_length=265):
#     layers = all_results[0].shape[1]
#     n_reps = len(all_results)
#     #Similarity
#     sim_ci_lower = []
#     sim_ci_upper = []
#     for l in range(layers):
#         v_l = []
        
#         if original:
#             for i in range(n_reps):
#                 if len(all_results[0].shape) == 3:                    
#                     random_agent=np.random.randint(0,3)
#                     r1 = all_results[i][:, :, random_agent]
#                 else:
#                     r1 = all_results[i][:, :]
#                 r2  = np.ones(len(r1[:,0])).astype(bool)
#                 r2[story_length:] = False
#                 v_l.append(jaccard_similarity(r1[:, l], r2))
#         else:
#             for i in range(n_reps):
#                 for j in range(i+1, n_reps):
#                     if len(all_results[0].shape) == 3:                    
#                         random_agent_1=np.random.randint(0,3)
#                         random_agent_2=np.random.randint(0,3)
#                         r1 = all_results[i][:, :, random_agent_1]
#                         r2 = all_results[j][:, :, random_agent_2]
#                     else:
#                         r1 = all_results[i][:, :]
#                         r2 = all_results[j][:, :]
                    
#                     v_l.append(jaccard_similarity(r1[:, l], r2[:, l]))
#         sim_ci=st.t.interval(alpha=0.95, df=len(v_l)-1, loc=np.mean(v_l), scale=st.sem(v_l))
#         sim_ci_lower.append(sim_ci[0])
#         sim_ci_upper.append(sim_ci[1])

#     return sim_ci_lower,sim_ci_upper


# def simulate_chain(prob, beta_n_observed=model_data.beta_n_observed, layers=6, story_length=1000, seed_length=265, introduction_rate=0.05):
#     #Chain
#     story = np.ones((story_length, layers)).astype(bool)

#     # First layer - transmission threshold defined by baseline transmission probabilities
# #     print(len(prob))
#     story[:, 0] = (np.random.random(story_length)<prob)

#     # Set newly added elements to false
#     story[seed_length:, 0] = False 
    
#     # Check whether any new words are introduced
#     introductions = (np.random.random(story_length)<introduction_rate)
#     # Set original story elements to false
#     introductions[:seed_length] = False 
#     # Combine original and newly added words
#     story[:, 0] |= introductions

#     for i in range(1, layers):
#         # Compute transmission probability
#         p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=story[:, i-1].sum(), network=0, n_remembered=1,story_length=seed_length)
        
#         # Check whether any new words are introduced
#         introductions = (np.random.random(story_length)<introduction_rate)
#         # Set original story elements to false
#         introductions[:seed_length] = False
        
#         # Update stories with transmitted words
#         story[:, i] = (story[:, i-1] & (np.random.random(story_length)<p)) | introductions
        
#     return story

        
# def simulate_net(prob, beta_n_observed=model_data.beta_n_observed, layers=6, story_length=1000, seed_length=265, introduction_rate=0.05):
#     #Network
#     story_n = np.ones((story_length, layers, 3)).astype(bool)
    
#     # First layer - transmission threshold defined by baseline transmission probabilities
#     story_n[:, 0, 0] = (np.random.random(story_length)<prob) 
#     story_n[:, 0, 1] = (np.random.random(story_length)<prob) 
#     story_n[:, 0, 2] = (np.random.random(story_length)<prob) 
#     # Set newly added elements to false
#     story_n[seed_length:,0,:] = False

#     # Check whether any new words are introduced for each participant in a generation
#     for r in range(3):
#         introductions = (np.random.random(story_length)<introduction_rate)
#         introductions[:seed_length] = False
#         # Combine original and newly added words
#         story_n[:, 0, r] |= introductions

#     for i in range(1, layers):
#         # Check how many participants remembered words in the previous generation
#         keep1 = story_n[:, i-1, 0]
#         keep2 = story_n[:, i-1, 1]
#         keep3 = story_n[:, i-1, 2]
#         sum_keep = story_n[:, i-1, :].sum(1)

#         # Loop over participants in the current generation
#         for j in range(3):
#             # Compute transmission probabilities 
#             # Note n_words divided by 3 because correct_odds assumes the newtork reads 3 times more words
#             p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=(sum_keep).sum()/3, network=1, n_remembered=sum_keep/3) 
#             # Determine whether words are kept according to union of individual participants in previous generation
#             keep = keep1|keep2|keep3 
#             keep &= np.random.random(story_length)<p
#             # Check for newly introduced words
#             introductions = (np.random.random(story_length)<introduction_rate)
#             introductions[:seed_length] = False
#             keep |= introductions

#             # Update stories with transmitted words
#             story_n[:, i, j] = keep
        
#     return story_n

# from dataclasses import dataclass
# @dataclass
# class ModelData:
#     beta_n_observed: float #increase with fraction of previous stories with the word
#     beta_network: float #fixed effect on network condition
#     beta_n_words: float #decay on number of words read
#     story_length: int 

# # This comes from the R code
# model_data = ModelData(
#     beta_n_observed=1.337,
#     beta_network=0.59321,
#     beta_n_words=-0.14566,
#     story_length=265
# )

# # def test():
# #     print("hello")
# #     return 0

# # def simulate_experiment(baseline_probs, layers = 6, seed_length=265, n_reps = 100, extra_w = 200, introduction_rate = 0.05, kept_rate = 0.1, beta_n_observed=model_data.beta_n_observed):
    
# # #     base_freqs = pd.read_csv(path_survival_analysis, sep="\t")
# # #     # GENERATE BASELINE TRANSMISSION PROBABILITIES FROM A BOUNDED EXPONENTIAL FITTED FROM THE DATA
# # #     loc, scale = sp.stats.expon.fit(base_freqs["baseline_prob_3_word_265"], floc=0)
# # #     np.random.seed(2018)
# # #     prob = bdd_exponential(story_length_=model_data.story_length,decay=1/scale,minval=0,maxval=0.95)
    
# # #     ## ADD IN EXTRA WORDS
# # #     prob = np.concatenate([baseline_probs, kept_rate+np.zeros(extra_w)])
# #     total_length = seed_length + extra_w

# #     # define error bar capsize
# #     cs=10

# #     # Define transmission computation type
# #     type_ = "integrate"

# #     all_results_n = []
# #     all_results_c = []

# #     # Run simulation
# #     for i in range(n_reps):

# #         sc = simulate_chain(prob,beta_n_observed=beta_n_observed, story_length=total_length,layers=layers,introduction_rate=introduction_rate)
# #         sn = simulate_net(prob,beta_n_observed=beta_n_observed, story_length=total_length,layers=layers,introduction_rate=introduction_rate)
# #         all_results_n.append(sn)
# #         all_results_c.append(sc)

# #     results_n = np.zeros((n_reps,total_length))
# #     results_c = np.zeros((n_reps,total_length))
# #     words_n = np.zeros((n_reps,layers))
# #     words_c = np.zeros((n_reps,layers))

# #     # Calculate similarities
# #     for i in range(n_reps): 
# #         results_n[i,:] = all_results_n[i][:, -1, :].mean(1)
# #         results_c[i,:] = all_results_c[i][:, -1]
# #         words_n[i,:] = all_results_n[i].mean(2).sum(0)
# #         words_c[i,:] = all_results_c[i].sum(0)

# #     results_n = results_n.mean(0)
# #     results_c = results_c.mean(0)

# #     fig1 = plt.figure(figsize=(5,3))
# #     fig2 = plt.figure(figsize=(5,3))
# #     fig3 = plt.figure(figsize=(5,3))
# #     fig4 = plt.figure(figsize=(5,3))


# #     ylabel=True
# #     ax1 = fig1.add_subplot(111)
# #     ax1.plot(prob,results_n,".",label="Network",color=cond2color["Network"])
# #     ax1.plot(prob,results_c,".",label="Chain",color=cond2color["Chain"])
# #     ax1.legend()
# #     ax1.set_xlabel("Probability of propagation")
# #     if ylabel:
# #         ax1.set_ylabel(f"Word frequency in layer {layers}")
# #     ax1.plot([0,1],[0,1],"--",color="lightgray")
# #     sns.despine(ax=ax1, bottom=True, left=True)
# #     ax1.grid(axis="y")

# #     ax2 = fig2.add_subplot(111)
# #     ax2.plot(range(layers),words_n.mean(0),label="Network",color=cond2color["Network"])
# #     ax2.plot(range(layers),words_c.mean(0),label="Chain",color=cond2color["Chain"])
# #     ax2.legend()
# #     ax2.set_xlabel("Generation")
# #     if ylabel:
# #         ax2.set_ylabel("Number of words")
# #     sns.despine(ax=ax2, bottom=True, left=True)
# #     ax2.grid(axis="y")

# #     sim_n = calculate_similarity(all_results_n) #, fun="std")
# #     sim_c = calculate_similarity(all_results_c) #, fun="std")
# #     sim_n_ci_lower,sim_n_ci_upper=calculate_sim_ci(all_results_n)
# #     sim_c_ci_lower,sim_c_ci_upper=calculate_sim_ci(all_results_c)
# #     yerr_n=np.empty((2,layers))
# #     yerr_n[0,:]=sim_n-np.array(sim_n_ci_lower)
# #     yerr_n[1,:]=np.array(sim_n_ci_upper)-sim_n
# #     yerr_c=np.empty((2,layers))
# #     yerr_c[0,:]=sim_c-np.array(sim_c_ci_lower)
# #     yerr_c[1,:]=np.array(sim_c_ci_upper)-sim_c

# #     ax3 = fig3.add_subplot(111)
# #     ax3.plot(range(1, layers+1),sim_n,label="Network",marker='o',color=cond2color["Network"])
# #     ax3.fill_between(range(1, layers+1), y1=sim_n_ci_lower,y2=sim_n_ci_upper, alpha=0.3)
# #     ax3.plot(range(1, layers+1),sim_c,label="Chain",marker='o',color=cond2color["Chain"])
# #     ax3.fill_between(range(1, layers+1), y1=sim_c_ci_lower,y2=sim_c_ci_upper, alpha=0.3)
# #     ax3.legend()
# #     ax3.set_xlabel("Generation")
# #     if ylabel:
# #         ax3.set_ylabel("Similarity between\nindependent replicates")
# #     sns.despine(ax=ax3, bottom=True, left=True)
# #     ax3.grid(axis="y")

# #     sim_n = calculate_similarity(all_results_n, original=True) #, fun="std")
# #     sim_c = calculate_similarity(all_results_c, original=True) #, fun="std")
# #     sim_n_ci_lower,sim_n_ci_upper=calculate_sim_ci(all_results_n, original=True)
# #     sim_c_ci_lower,sim_c_ci_upper=calculate_sim_ci(all_results_c, original=True)
# #     yerr_=np.empty((2,layers))
# #     yerr_n[0,:]=sim_n-np.array(sim_n_ci_lower)
# #     yerr_c[1,:]=np.array(sim_n_ci_upper)-sim_n

# #     ax4 = fig4.add_subplot(111)
# #     ax4.plot(range(1, layers+1),sim_n,label="Network",marker='o',color=cond2color["Network"])
# #     ax4.fill_between(range(1, layers+1), y1=sim_n_ci_lower,y2=sim_n_ci_upper, alpha=0.3)
# #     ax4.plot(range(1, layers+1),sim_c,label="Chain",marker='o',color=cond2color["Chain"])
# #     ax4.fill_between(range(1, layers+1), y1=sim_c_ci_lower,y2=sim_c_ci_upper, alpha=0.3)
# #     ax4.legend()
# #     ax4.set_xlabel("Generation")
# #     if ylabel:
# #         ax4.set_ylabel("Similarity with the \noriginal story")
# #     sns.despine(ax=ax4, bottom=True, left=True)
# #     ax4.grid(axis="y")


# #     # fig1.savefig(f"{figures_path}sim_cm1.pdf", bbox_inches="tight")   
# #     # fig3.savefig(f"{figures_path}sim_cm2.pdf", bbox_inches="tight")   
# #     fig3.savefig(f"{path_figures}sim_cm3.pdf", bbox_inches="tight")   
# #     fig4.savefig(f"{path_figures}sim_cm4.pdf", bbox_inches="tight")  







