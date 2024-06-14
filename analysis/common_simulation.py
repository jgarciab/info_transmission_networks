from common_variables import *
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st

from scipy.stats import binom 


def create_prob(prob, beta_n_observed=model_data.beta_n_observed, beta_network=model_data.beta_network,
    n_remembered=1, network=0, n_words=265, beta_n_words=model_data.beta_n_words,
    chain_less_words_factor=3,story_length=model_data.story_length):
    """
    Adjusts the given probability value using a statistical model, assuming all variables stay constant except the parameters below. 
    Takes as baseline the baseline probability.
    
    Parameters:
    - prob (float): p_{j,1} - Input probability value to be adjusted. Should be between 0 and 1.
    - beta_network (float, optional): beta_1 - parameter for reading the story several times.
    - beta_n_observed (float, optional): beta_2 - parameter for reading the story several times. Default is 1.337.
    - beta_n_words (float, optional): beta_3 - parameter for number of words read. Default is 1.5681.
    - network (int, optional): C - Indicates whether the data is in a network (1) or not (0). Default is 0.
    - n_words (int, optional): n_w - Number of words in the original text. Default is 265.
    - n_remembered (float, optional): N - Fraction of times the event is remembered. Should be between 0 and 1. Default is 1.
    - story_length (int, optional): sum_j w_{j,k}: current number of words in story
    
    Returns:
    - float: Adjusted probability value.
    """
    # np.log(prob / (1 - prob)) = (
    #     alpha_0 +
    #     beta_n_observed * 1 +                   # Adjust based on how manytimes an event is remembered
    #     beta_n_words * np.log(story_length)     # Adjust based on the number of words
    # )

    # Get alpha0 from (equation above)
    alpha_0 = np.log(prob / (1 - prob)) - beta_n_observed - beta_n_words * np.log(story_length)
    
    # Apply corrections to the log odds using the statistical model
    odds = np.exp(
     alpha_0 
     + beta_n_observed * n_remembered           # Adjust based on how many times an event is remembered
     + network * beta_n_words * np.log(chain_less_words_factor) +  # Adjust for being in a network, assuming 3x more words are read
     + network * beta_network
     + beta_n_words * np.log(n_words)     # Adjust based on the numberof words
    )
    
    # Convert the corrected odds back to a probability
    adjusted_prob = odds / (odds + 1)

    return adjusted_prob

def create_probs(prob, beta_n_observed=model_data.beta_n_observed, n_words=265, network=1, story_length=265):
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
    
    base_freq = pd.read_csv(path_survival_analysis, sep="\t")
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
    prob = np.random.random(story_length)
    return prob


def fitted_exponential(story_length_,minval=0,maxval=0.9):
    """
    Generates a list of bounded exponential variable with decay rate fitted from the data.
    
    Parameters:
    - story_length_ (int): number samples to produce
    - decay (float): decay rate of exponential RV
    - minval (int): minimum of exponential distribution
    - maxval (int): maximum of exponential distribution
    
    Returns:
    list of probabilities.
    """
    base_freqs=get_base_freqs()
    loc, scale = sp.stats.expon.fit(base_freqs, floc=0)
    prob=bdd_exponential(story_length_=story_length_,decay=1/scale,minval=minval,maxval=maxval)
    return prob


def fitted_lognormal(story_length_=100,maxval=0.9):
    base_freqs=get_base_freqs()
    shape, loc, scale = sp.stats.lognorm.fit(base_freqs, loc=0)
    lognmu, lognsigma=np.log(scale), shape
    prob=np.random.lognormal(mean=lognmu,sigma=lognsigma,size=10*story_length_)
    prob=prob[prob<maxval] #Truncate values below a threshold (note in general they can be sampled as prob>1)
    prob = np.random.choice(prob, size=story_length_, replace=False)
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
        # jaccard_sim=2*np.sum((a&b))/265
    return jaccard_sim

def calculate_similarity(all_results, original=False, fun="mean",story_length=265):
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
                r2  = np.ones(len(r1[:,0])).astype(bool)
                r2[story_length:] = False
                v_l.append(jaccard_similarity(r1[:, l], r2))
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
    return sim


def calculate_sim_ci(all_results, original=False, story_length=265):
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
                r2  = np.ones(len(r1[:,0])).astype(bool)
                r2[story_length:] = False
                v_l.append(jaccard_similarity(r1[:, l], r2))
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
    #Chain
    story = np.ones((story_length, layers)).astype(bool)

    # First layer - transmission threshold defined by baseline transmission probabilities
#     print(len(prob))
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
        # Compute transmission probability
        p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=story[:, i-1].sum(), network=0, n_remembered=1,story_length=seed_length)
        
        # Check whether any new words are introduced
        introductions = (np.random.random(story_length)<introduction_rate)
        # Set original story elements to false
        introductions[:seed_length] = False
        
        # Update stories with transmitted words
        story[:, i] = (story[:, i-1] & (np.random.random(story_length)<p)) | introductions
        
    return story

        
def simulate_net(prob, beta_n_observed=model_data.beta_n_observed, layers=6, story_length=1000, seed_length=265, introduction_rate=0.05):
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
            p = create_prob(prob, beta_n_observed=beta_n_observed, n_words=(sum_keep).sum()/3, network=1, n_remembered=sum_keep/3) 
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
    import seaborn as sns
    import pylab as plt
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

