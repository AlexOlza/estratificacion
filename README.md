# estratificacion

# Description

The stratification program in the Basque Country classifies the population based on their expected healthcare needs to provide proactive interventions. Using morbidity variables derived from clinical records, we develop predictive models to foresee unwanted health complications such as unplanned hospitalization or high healthcare expenditures, interpreted as a proxy for increased healthcare needs. It is an ongoing project led by Osakidetza with the collaboration of BCAM.

In this repo, we focus on predicting the probability of unplanned hospitalization and compare several statistical learning techniques and several groups of predictor variables.

# Disclaimer 

**The data is confidential and will never be made available.**
However, we are allowed to describe it, and think our code could be of use for people working with similar datasets.
In our case, we extract most of our predictor variables from the ACG System (Johns Hopkins, not open-source, extensively used in the literature). This case-mix system processes information from the clinical history and generates a collection of categorical variables based on all the diagnosis, prescription medications, and additional information for each patient. Their algorithm is private. 

We are also starting to incorporate resource usage variables, extracted directly from Osakidetza sources. These variables contain information such as the number of primary care or ER visits, wether the patient is on dialisis or has had any major procedure during the past year, etc.

# Usage- Repo Structure

Explanation of the repository structure [here](https://github.com/AlexOlza/estratificacion/blob/2ff68b6833a2b7676cbc2c522a9f20c0584ba7d9/docs/structure.md).

