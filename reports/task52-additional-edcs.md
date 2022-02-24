# ABOUT THE PREDICTIVE VALUE OF ACG GROUPS

The ACG comorbidity groups are the only variable we can't construct from Osakidetza's Electronic Health Records- We need Johns Hopkins' proprietary software to extract them. In this document we assess the contribution of this variable. To do so, we build a series of nested models.

## Variable description:
- EDC: Diagnosis
- RXMG: Pharmacy prescriptions
- ACG: Adjusted Clinical Groups

## Experiment description:
By "experiment" we mean a specific data collection process. Thus, the chosen experiment completely determines the DataFrames `X` (predictors) and `y` (response). Below are the definitions of the experiments we conducted to assess the value of the ACG tool:

- `cost`: The response is the **total healthcare cost**. We exclude OSIs 16 and 22 (Errioxa and Tolosaldea). We use the EDC and RxMg codes selected by ACG models.

- `cost_fulledcs`: Same as the former, but we use **all** the EDCs and RxMgs (282 EDCs and 77 RxMgs)

- `urgcms_excl_nbinj`: The response is **unplanned hospitalization** as defined by the algorithm **CMS**. We exclude OSIs 16 and 22 (Errioxa and Tolosaldea). We discount the admissions due to birth&delivery or traumatological injuries. We use the EDC and RxMg codes selected by ACG models.

- `urgcms_excl_nbinj_fulledcs`: Same as above but with complete EDC and RxMg codes.

-  `almeria`: We define unplanned admissions with administrative criteria, and we do not exclude OSIs 16 and 22 nor birth and delivery or traumatological injuries. We use the EDC and RxMg codes selected by ACG models.

## RESULTS I) Predicting healthcare expenditures:

Experimental framework `cost`. With such data we fit nested linear regression models. The score is R2. To compute the other metrics, we order the **observed** costs in 2018 and flag the first 20000. We do the same with the **predicted** costs. 

Note that recall is the same as PPV except in the first model. This is expected, because FP=FN! This arises from the fact that we have artificially defined our list, with 20000 "positives" in the observed data (TP+FN=20000) and also 20000 "positives" among our predictions (TP+FP=20000). 
The first model does not exhibit this, because due to lack of flexibility (too few parameters) it selects a longer list: The top 31114 predictions have the same value. 


| Model       | Predictors   |     Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|----------:|---------------:|------------:|
| nested\_lin2 | SEX+ AGE     | 0.0572203 |        0.0609  |   0.0391464 |
| nested\_lin4 | + EDC\_       | 0.183223  |        0.2209  |   0.2209    |
| nested\_lin3 | + RXMG\_      | 0.220087  |        0.2384  |   0.2384    |
| nested\_lin1 | + ACG        | 0.223368  |        0.23775 |   0.23775   |

In the experiment `cost_fulledcs`, the results are:

| Model       | Predictors   |     Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|----------:|---------------:|------------:|
| nested\_lin1 | SEX+ AGE     | 0.0572203 |         0.0609 |   0.0391464 |
| nested\_lin2 | + EDC\_       | 0.195939  |         0.2084 |   0.2084    |
| nested\_lin3 | + RXMG\_      | 0.212433  |         0.2203 |   0.2203    |
| nested\_lin4 | + ACG        | 0.218453  |         0.2214 |   0.2214    |

**INTERPRETATION:** Forcing the use of the codes discarded by the ACG system worsens linear the models. We could inspect whether this also happens with a nonlinear algorithm (we could try a 2 layer neural network with default hyperparameters and see how that goes).

## RESULTS II) Predicting unplanned hospitalization:

Experimental framework `urgcms_excl_nbinj`. With such data we fit nested logistic regression models. The score is AUC.

| Model       | Predictors   |    Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|---------:|---------------:|------------:|
| nested\_log1 | SEX+ AGE     | 0.731795 |      0.0532175 |    0.251262 |
| nested\_log2 | + EDC\_       | 0.773601 |      0.0753489 |    0.4792   |
| nested\_log3 | + RXMG\_      | 0.791868 |      0.0774637 |    0.49265  |
| nested\_log4 | + ACG        | 0.798397 |      0.0777625 |    0.49455  |

With data from `urgcms_excl_nbinj_fulledcs`, the results are:

| Model       | Predictors   |    Score |   Recall\_20000 |   PPV\_20000 |
|:------------|:-------------|---------:|---------------:|------------:|
| nested\_log1 | SEX+ AGE     | 0.731795 |      0.0532175 |    0.251262 |
| nested\_log2 | + EDC\_       | 0.798865 |      0.0752938 |    0.47885  |
| nested\_log3 | + RXMG\_      | 0.802037 |      0.0772043 |    0.491    |
| nested\_log4 | + ACG        | 0.805559 |      0.0776524 |    0.49385  |

**INTERPRETATION:** Using full EDC and RxMg codes slightly increases the AUCs, but does not improve the metrics in the *K=20000* list of patients. Thus, the improvement in performance is not happening among those with highest predicted probabilities. We need to plot the ROC curve for further analysis.

## Additional variables and reproducibility:

**This is a comparison between experiments**, not between nested models.
Looking at respective `nested_log4`models, we can also inspect the effect of HOSDOM|FRAILTY|INGRED\_14GT in each experimental framework. We rebuild the logistic model from Almería to check reproducibility (achieved!). Predictors: SEX+ AGE+ EDC+ RXMG + ACG +HOSDOM +FRAILTY+ INGRED\_14GT in all models.


| Model                    |           Experiment        |    Score |  Recall\_20000 |  PPV\_20000 |
|:-------------------------|:----------------------------|---------:|---------------:|------------:|
| logistic20220207\_122835 |  `urgcms_excl_nbinj`        | 0.798511 |      0.0775502 |      0.4932 |
| logistic20220223\_170519  | `urgcms_excl_nbinj_fulledcs`| 0.805557 |      0.0777625 |     0.49455 |
| logistic20220210\_165040 | `almeria`                   |  0.80507 |        0.10144 |      0.5175 |
