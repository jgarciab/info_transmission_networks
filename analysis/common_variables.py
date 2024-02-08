from pathlib import Path

transformer_model = 'sentence-transformers/all-mpnet-base-v2'

# Path data
path_data_files = "../data/"

# Where to save the text models (very large)
path_model = Path("~/tmp").expanduser()
path_model.mkdir(parents=True, exist_ok=True)

# Path where to save figures (e.g. overleaf)
path_figures = "/Users/garci061/Dropbox/Apps/Overleaf/2022_rumor_final/Figures/"# "./results"

# Path of experiment data
path_final_data = f'{path_data_files}/final_experiment/full_data.csv'
path_text_embeddings = f'{path_data_files}/text_embeddings/'

path_survival_analysis = f'{path_data_files}/data_final/word_baseline_probs.csv'

## Compare layer 1 to this layer in the analysis
comparison_layer = 6



# Coefficients of the R model
from dataclasses import dataclass
@dataclass
class ModelData:
    beta_n_observed: float #increase with fraction of previous stories with the word
    beta_network: float #fixed effect on network condition
    beta_n_words: float #decay on number of words read
    story_length: int 

# This comes from the R code
model_data = ModelData(
    beta_n_observed=1.337,
    beta_network=0.59321,
    beta_n_words=-0.14566,
    story_length=265
)



story_original = "Through history, most people didn't die of cancer or heart disease, the lifestyle diseases that are common in the Western world today. This is mostly because people didn't live long enough to develop them. They died of injuries -- being gored by an ox, shot on a battlefield, crushed in one of the new factories of the Industrial Revolution -- and most of the time from infection, which followed those injuries.\n\nThat changed when antibiotics were discovered. In 1928, Alexander Fleming discovered penicillin, a drug still used today to fight bacterial infections. Suddenly, infections that had been a death sentence became remedied within days. During World War II, this drug treated pneumonia and sepsis, and has been estimated to have saved between 12-15% of Allied Forces lives. We have been living inside the golden epoch of the miracle drugs, and now, we are coming to an end of it.\n\nPeople are dying of infections again because of a phenomenon called antibiotic resistance, or popularly referred to as “superbugs”. Bacteria compete against each other for resources, for food, by manufacturing lethal compounds that they direct against each other. When we first made antibiotics, we took those compounds into the lab and made our own versions of them, and bacteria responded to our attack the way they always had.\n\nFor 70 years, we played a game of leapfrog -- our drug and their resistance, and then another drug, and then resistance again -- and now the game is ending. Bacteria develop resistance so quickly that pharmaceutical companies have decided making antibiotics is not in their best interest, so there are infections moving across the world for which, out of the more than 100 antibiotics available on the market, two drugs might work with side effects, or one drug, or none.\n\nIt would be natural to hope that these infections are extraordinary cases, but in fact, in the United States and Europe, 50 thousand people a year die of infections which no drugs can help. A project chartered by the British government known as the Review on Antimicrobial Resistance estimates that the worldwide toll right now is 700 thousand deaths a year. Also, if we can't get this under control by 2050, the worldwide toll will be 10 million deaths a year (more than the current population of New York City).\n\nThe scale of antibiotic resistance seems overwhelming, but if you've ever bought a fluorescent light bulb because you were concerned about climate change, you already know what it feels like to take a tiny step to address an overwhelming problem. We could take those kinds of steps for antibiotic use too. We could forgo giving an antibiotic for our kids’ ear infection, if we're not sure it's the right one. And we could promise each other to never again to buy chicken or shrimp or fruit raised with routine antibiotic use. If we did those things, we could slow down the arrival of the post-antibiotic world."
story_original = story_original.replace("\n"," ")


# nltk.download('stopwords')
# from nltk.corpus import stopwords
# eng_stopwords = set(stopwords.words("english"))
eng_stopwords = {'myself', 'm', "isn't", "wasn't", 'ours', 'having', 'itself', 'won', 'himself', 'd', "you've", 'those', 'or', 'be', 'them', 'hadn', 'mightn', "don't", 'it', "couldn't", 'this', 'but', 'me', 'he', 'because', 'at', "it's", 'over', 'than', 'does', 'll', "mightn't", 'do', "should've", "you're", 'have', 'here', 'hers', 'very', 'hasn', 'yours', 's', 'if', 'too', 'their', 'again', 'during', 'a', 'did', 'will', 'needn', 'shouldn', 'what', 'wasn', 'didn', 'until', 'being', 'to', 'in', 'just', 'aren', "haven't", 'below', 'themselves', 'by', 'more', "weren't", 'no', 'above', 'has', "hadn't", 'y', 'from', 'ma', 'ourselves', 'between', "doesn't", "didn't", 'been', 'such', 't', 'are', 'with', 'after', 'as', 'out', 'was', 'o', 'any', 'why', 'that', 'most', 'once', 'haven', 'my', 'an', 'while', 'further', 'is', 'wouldn', 'herself', 'ain', 'there', 'nor', 'under', 'whom', 'few', 'shan', 'its', 'our', 'her', "hasn't", 'both', 'each', 'own', 'through', 'and', 'all', 'only', 'you', 'yourselves', 'when', "shan't", "she's", 'against', 'before', 'now', 've', 'yourself', 'weren', 'of', 're', 'his', "needn't", 'for', 'these', 'were', 'about', 'mustn', 'which', "that'll", 'down', 'into', 'where', 'same', 'can', 'isn', 'she', 'on', 'up', 'i', 'other', 'your', 'theirs', 'so', 'some', "wouldn't", 'the', "you'll", 'had', 'they', 'doing', "won't", 'how', "shouldn't", 'we', 'him', "aren't", 'couldn', 'off', 'not', "you'd", 'doesn', 'then', 'don', 'should', 'am', "mustn't", 'who'}