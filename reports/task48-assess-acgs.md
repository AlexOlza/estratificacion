# ABOUT THE PREDICTIVE VALUE OF ACG GROUPS

The ACG comorbidity groups are the only variable we can't construct from Osakidetza's Electronic Health Records- We need Johns Hopkins' proprietary software to extract them. In this document we assess the contribution of this variable. To do so, we build a series of nested models.

## Variable description:
- EDC: Diagnosis
- RXMG: Pharmacy prescriptions
- ACG: Adjusted Clinical Groups

## Predicting healthcare expenditures:

Here we present nested models within the experimental framework `cost`, which determines the response variable. We exclude OSIs 16 and 22 (Errioxa and Tolosaldea). With such data we fit nested linear regression models. The score is R2. To compute the other metrics, we order the **observed** costs in 2018 and flag the first 20000. We do the same with the **predicted** costs. 

Note that recall is the same as PPV except in the first model. This is expected, because FP=FN! This arises from the fact that we have artificially defined our list, with 20000 "positives" in the observed data (TP+FN=20000) and also 20000 "positives" among our predictions (TP+FP=20000). 

The first model does not exhibit this, because due to lack of flexibility (too few parameters) it selects a longer list: The top 31114 predictions have the same value. 


| Model       | Predictors   |     Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|----------:|---------------:|------------:|
| nested\_lin2 | SEX+ AGE     | 0.0572203 |        0.0609  |   0.0391464 |
| nested\_lin4 | + EDC\_       | 0.183223  |        0.2209  |   0.2209    |
| nested\_lin3 | + RXMG\_      | 0.220087  |        0.2384  |   0.2384    |
| nested\_lin1 | + ACG        | 0.223368  |        0.23775 |   0.23775   |

## Predicting unplanned hospitalization:

Here we present nested models within the experimental framework `urgcms_excl_nbinj`, meaning that we consider planned admissions as defined by the CMS algorithm and we exclude those related to birth and delivery or traumatological injuries, but we keep day hospital admissions. We also exclude OSIs 16 and 22 (Errioxa and Tolosaldea). With such data we fit nested logistic regression models. The score is AUC.

| Model       | Predictors   |    Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|---------:|---------------:|------------:|
| nested\_log1 | SEX+ AGE     | 0.731795 |      0.0532175 |    0.251262 |
| nested\_log2 | + EDC\_       | 0.773601 |      0.0753489 |    0.4792   |
| nested\_log3 | + RXMG\_      | 0.791868 |      0.0774637 |    0.49265  |
| nested\_log4 | + ACG        | 0.798397 |      0.0777625 |    0.49455  |

## Additional variables and reproducibility:

We inspect the effect of HOSDOM|FRAILTY|INGRED\_14GT in the experimental framework `urgcms_excl_nbinj`. We rebuild the logistic model from Almeria to check reproducibility (achieved!). The only differences betwen the models below arise from the data (`urgcms_excl_nbinj` vs `almeria`). 

In `almeria`, we define unplanned admissions with administrative criteria, and we do not exclude OSIs 16 and 22 nor birth and delivery or traumatological injuries. The table shows that, in fact, we have losed performance. WHY??

| Model                            |           Predictors         |    Score |   Recall\_20000 |   PPV\_20000 |
|:---------------------------------|:-----------------------------|---------:|---------------:|------------:|
| logistic20220207\_122835          | +HOSDOM|FRAILTY|INGRED\_14GT  | 0.798511 |      0.0775502 |      0.4932 |
|:---------------------------------|:-----------------------------|---------:|---------------:|------------:|
| logistic20220210\_165040 (almeria)| +HOSDOM|FRAILTY|INGRED\_14GT  |  0.80507 |        0.10144 |      0.5175 |
