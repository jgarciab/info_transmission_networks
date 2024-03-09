Replication code for paper `Redundancy in transmission promotes information consensus across independent social networks`.

# Citation
Paper (arxiv for now) + code/data (zenodo DOI)


# Replication of the analysis
- Download data from Zenodo --> xxx
- Create a conda environment using environment.yml: `conda env create -f environment.yml`
- Run notebooks in `analysis` folder using the `rumor` environment that you just created


# Replication of the experiment

The experiment was run using [Dallinger](https://dallinger.readthedocs.io/en/latest). Code for replicating the experiment is available in the experiment_mturk/dallinger_experiment/ directory. To test an experiment locally, navigate to the experiment directory, and then run: 

```
$ pip install -r requirements.txt
$ dallinger debug --verbose
```

For debugging, you may want to reduce `num_replications` in `experiment.py` as each recruitment request will open a new browser window. Also note that installing Dallinger can be buggy and there are often version conficts between required packages.