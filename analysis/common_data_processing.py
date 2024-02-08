from common_variables import *
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap
from collections import Counter

## CLEAN UP THE HEROKU DATA

def process_file(data, out_path, chain=True, story_original=story_original):
    """
    Processes and cleans up data from a Heroku database, then writes the cleaned data to a file.

    This function iterates over groups of data segmented by replication factor. For each group,
    it sorts the data by generation, then formats and writes the data to an output file in a tab-separated format.
    The output includes the generation number, replication factor, three versions of the story based on generation logic,
    and the merged story.

    Parameters:
    - data (DataFrame): The input data to be processed. Expected to be a pandas DataFrame with columns
      'replication', 'generation', and 'response', among others.
    - out_path (str): The file path where the cleaned data should be written. The data is saved in a tab-separated format.
    - chain (bool, optional): Determines whether experimental design is a chain. If True,
      the previous generation's response is used for all three story versions in the current generation.
      If False, it uses distinct responses from the previous generation for each story version. Default is True.

    Returns:
    None. The function writes the cleaned data to the specified file path.
    """

    
    
    # Open or create the file at out_path for writing
    with open(out_path,"w+") as f:
        # Write the header row to the file
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("layer_n","rep","story1","story2","story3","story_merged"))

        # Iterate over each replication factor in the data
        for i, rep in data.groupby("replication"):
            # Sort the replication group by generation and iterate over each row
            for rn, row in rep.sort_values(by="generation").iterrows():

                gen = row["generation"]  # Current generation

                story_merged = row["response"]  # The merged story response
                # If this is the first generation, initialize all story versions to the original story
                if gen < 1:
                    story1, story2, story3 = story_original, story_original, story_original
                else:
                    # For subsequent generations, determine the story versions based on the 'chain' parameter
                    if chain:
                        # If chaining, use the previous generation's response for all story versions
                        pv = rep.loc[rep["generation"] == gen - 1, "response"].values[0]
                        story1, story2, story3 = pv, pv, pv
                    else:
                        # If not chaining, use distinct responses from the previous generation for each story version
                        story1, story2, story3 = rep.loc[rep["generation"] == gen - 1, "response"].values

                # Write the processed data to the file, replacing newline characters in stories to ensure format consistency
                f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(gen + 1, i, story1.replace("\n", " "), story2.replace("\n", " "), story3.replace("\n", " "), story_merged.replace("\n", " ")))

## CREATE EMBEDDINGS


def create_embeddings(results, transformer_model=transformer_model, path=f"{path_text_embeddings}/story_embeddings"):
    """
    From a dataframe with layer, rep, cond and story, embed using a transformer model and save
    """
    from sentence_transformers import SentenceTransformer, util
    import pickle
    # Add original and person ID (so we compare persons of different replications later on)
    data = results.loc[:, ["layer_n","rep","condition","story_merged"]]
    data.loc[9909] = [8,0,"Full",story_original]
    data["k"] = data.index%3
    data.loc[data["condition"].str.contains("Chain"),"k"] = 0
    
    # Open model
    model_t = SentenceTransformer(transformer_model) # Defined in common_variables.py
    
    # Embedd results
    emb = model_t.encode(data["story_merged"].values, convert_to_tensor=True)#, device="cuda") #if support for gpu
    
    # Tensors to pandas for saving
    emb_df = pd.DataFrame(np.array(emb))
    emb_df.loc[:,"condition"] = data["condition"].values
    emb_df.loc[:,"layer_n"] = data["layer_n"].values
    emb_df.loc[:,"k"] = data["k"].values
    emb_df.loc[:,"rep"] = data["rep"].values
    
    emb_df.to_csv(f"{path}.csv", index=None)
    np.save(f"{path}.npy", np.array(emb))

    # Read global_emb from the embeddings
    global_emb = dict(zip(data["story_merged"].values, np.array(emb)))
    pickle.dump(global_emb, open(f"{path}_global_emb.pickle", "wb+"))

    return emb

def project_embeddings(emb, path=f"{path_text_embeddings}/X_story_embedded_"):
    # Project to TSNE 
    X_embedded = TSNE(random_state=42, n_components=2, learning_rate='auto', n_jobs=-1).fit_transform(emb)
    np.save(f"{path}_tsne.npy", X_embedded)
    print(X_embedded.shape)
    
    # Project to UMAP
    X_embedded = umap.UMAP(random_state=42, metric="cosine").fit_transform(emb)
    np.save(f"{path}_umap.npy", X_embedded)
    print(X_embedded.shape)

