# Report of results from experiment `almeria`

The logistic model is well reproduced, but the rest are not. This may be related to some of these reasons:

- The random hyperparameter search -> Should be fixed by using the same random seeds
- Parallelization in training -> Unavoidable
- Internal randomness in sklearn methods -> Should be fixed by using the same random seeds

| Model                       | Predictors   |    Score |   Recall\_20000 |   PPV\_20000 |
|:----------------------------|:-------------|---------:|---------------:|------------:|
| HGB20220209\_133310          |              | 0.813178 |      0.105811  |      0.5398 |
| logistic20220210\_165040     |              | 0.80507  |      0.10144   |      0.5175 |
| randomForest20220208\_113412 |              | 0.804188 |      0.0890318 |      0.4542 |
