# Repository structure and basic usage concepts

## Configuration files

### Default (basic attributes, stored in `configurations/default.py`)
```
SEED #random seed
"""LOCATIONS"""
PROJECT, ROOTPATH, DATAPATH, INDISPENSABLEDATAPATH,CONFIGPATH, USEDCONFIGPATH to save configs, OUTPATH to save predictions, MODELSPATH to save models

"""FILES"""
ALLHOSPITFILE,
ACGFILES={2016:filename,
          2017:filename,
          2018:filename}
ACGINDPRIVFILES={same structure ...} With privation index
ALLEDCFILES={same structure ...}
FULLACGFILES={same structure ...}

VERBOSE, TRACEBACK=True or False 
```


### Experiments
They refer to the data collection step. The chosen experiment determines the inclusion criteria for patients (e.g. whether to exclude certain OSIs), the response variable (the definition of unplanned hospitalization, or cost) and the predictor variables. Some of the experiments currently in use are:

- `urgcms_excl_nbinj`: The response variable is unplanned hospitalization by CMS criteria. The predictors are ACGs, EDCs, RXMGs, Age, Sex, Hospital dominant diagnoses, frailty and having more than 14 prescription ingredients. We use only the EDC and ACG codes selected by the ACG software. We exclude patients from Tolosaldea and Errioxa because they receive
most of their care outside of Osakidetza. We discount the unplanned hospitalizations due to birth & delivery or traumatological injuries.
- ...

To define a new experiment (for example, to change the predictors), add a file named `configurations/experiment_name.py` with the text:

```
EXPERIMENT='experiment_name'
PREDICTORREGEX= Regex string r'...' that matches the columns you want to use as predictors. This may include previous hospitalizations.
INDICEPRIVACION= False or True ; Wether you want to use the privation index (beware, it has missing values)
COLUMNS=['...'] String with the name of the response variable. Examples: ['urgcms'], ['urg'], ['COSTE_TOTAL_ANO2']
EXCLUDE=['nbinj'] or ['hdia'] or [] or ['nbinj','hdia'] To exclude birth&delivery and/or day hospital admission, or none.
EXCLUDEOSI=['OS16','OS22'] To exclude those OSIs (Look up the code of the OSIs you want to exclude)
RESOURCEUSAGE=False or True, to include the number of consultations, dialisis, cancer treatment..... as predictions
```
All the fields are necessary and should be capitalized.

### Algorithms: 
They refer to the modelization process. They receive the chosen configuration name and the experiment name via command line arguments.
They import everything from the default configuration and from the experiment file,  using `importlib`.

```
MODELPATH=MODELSPATH+EXPERIMENT+'/' Always like this
USEDCONFIGPATH+=EXPERIMENT+'/' Always like this
ALGORITHM='randomForest' or 'logistic' or any other. If algorithm starts with "neural", Keras will be used.
CONFIGNAME='an identifiable name'
PREDPATH=os.path.join(OUTPATH,EXPERIMENT) Always like this
FIGUREPATH=os.path.join(ROOTPATH,'figures',EXPERIMENT) Always like this

TRACEBACK=False or True for debugging purposes, will show all the function calls and exits.

ESTIMATOR= an sklearn or Keras estimator, not necessarily named ESTIMATOR

... Anything extra, everything below is optional ....
... hyperparameter options ...
RANDOM GRID= A dictionary for hyperparameter tuning in sklearn
N_ITER=... 
CV= ...
```

## Main files
Coming soon...
