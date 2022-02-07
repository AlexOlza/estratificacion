# HOW TO SET-UP AN EXPERIMENT (WORK IN PROGRESS)

## Definitions:
- **Experiment:** It refers to the data configuration rather than the modelling process. The experiment determines a set of **predictors** (i.e. ACGs, EDCs,..., resource usage indicators), a **response** variable, **exclusion criteria** for patients (i.e. exclude certain OSIs or certain hospital admissions). 

- **Algorithm:** The modelling technique (i.e. logistic regression)

## Our use case:

We are comparing 3 algorithms (log.reg., RF, HGB) in each experiment (we will deal with Neural Networks later on). We have planned the following experiments:

1. **alm:** Almeria (to do)
2. **alm_eOSI:** Almeria but excluding patients from OSIs 16 and 22 (to do)
3. **cms_eHNB_eOSI:** Use CMS algorithm to determine unplanned admissions, and exclude patients from OSIs 16 and 22. Also exclude admissions to day hospitals and those due to birth & delivery or injuries
4. **cms_eNB_eOSI:** Same as 3, but do NOT exclude admissions to day hospitals

## Necessary files per experiment and their naming convention:
Let exp be an experiment and alg an algorithm. You need to provide:

1. If running in the cluster: 
   - `main/cluster/exp.py`
   - `main/cluster/exp.sl` , with the line `spython exp.py alg exp` 
   - `main/cluster/alg.py` 
   - `configurations/cluster/alg.py` 

2. If running locally: ADD EXPLANATION




