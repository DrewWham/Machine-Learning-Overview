# Overview-of-Machine-Learning

## Languages, Software Libraries and Packages 
### Base Languages
* [R](https://www.r-project.org)
* [Python](https://www.python.org)

### Data Wrangling and Transformation
* [datatables](https://github.com/Rdatatable/data.table/wiki/Getting-started)([cheetsheet](https://s3.amazonaws.com/assets.datacamp.com/img/blog/data+table+cheat+sheet.pdf))
* [dplyr](http://dplyr.tidyverse.org)([cheetsheet](https://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf))
* [reshape2](https://cran.r-project.org/web/packages/reshape2/reshape2.pdf)([cheetsheet](http://rstudio-pubs-static.s3.amazonaws.com/14391_c58a54d88eac4dfbb80d8e07bcf92194.html))
* [pandas](https://pandas.pydata.org)([cheetsheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf))

### Cluster Computing
* [pyspark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) ([cheetsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf))
* [sparkr](https://spark.apache.org/docs/latest/sparkr.html)
* [H20](https://www.h2o.ai)

### Cloud Computing
* [AWS](https://aws.amazon.com)

### Modeling
* [Tensorflow](https://www.tensorflow.org)
* [Pytorch](http://pytorch.org)
* [H2O](https://www.h2o.ai)

## ML Algorithms
--------
### Supervised
--------
Use cases: Prediction, classification and labeling, quntification of risk and uncertinty, feedback based recomendation
Examples: Sales forecasting, Vegas odds, insurance risk, credit fraud detection

#### Regression
* Least Squares
* Sequential
* Penalized Regression - [glmnet - R Package](https://cran.r-project.org/web/packages/glmnet/glmnet.pdf)
   * Ridge
   * LASSO
   * Group LASSO, Elastic Net ...
* Gaussian Process regression - [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) [Python code](http://scikit-learn.org/stable/modules/gaussian_process.html)
* Decision Tree - ([R package - Generalized Boosted Regression Model](https://cran.r-project.org/web/packages/gbm/gbm.pdf)) ([ R package - random forest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)) ([Python package with GPU acceleration - lightGBM ](http://lightgbm.readthedocs.io/en/latest/)) ([R and Python package with GPU acceleration - xgboost](http://xgboost.readthedocs.io/en/latest/))
* [Support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine) ([GPU accelerated code](https://github.com/zeyiwen/thundersvm))
* Neural Networks 
#### Classification
* K-Nearest Neighbors - [Fast KNN](https://cran.r-project.org/web/packages/FNN/FNN.pdf)
* Decision Tree - ([R package - Generalized Boosted Regression Model](https://cran.r-project.org/web/packages/gbm/gbm.pdf)) ([ R package - random forest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)) ([Python package with GPU acceleration - lightGBM ](http://lightgbm.readthedocs.io/en/latest/)) ([R and Python package with GPU acceleration - xgboost](http://xgboost.readthedocs.io/en/latest/))

#### Feature Importance
* Factor Analysis
* Analysis of Feature Importance - [XGBoost explainer](https://medium.com/applied-data-science/new-r-package-the-xgboost-explainer-51dd7d1aa211) ([R Package](https://github.com/AppliedDataSciencePartners/xgboostExplainer)) ([randomForestExplainer](https://cran.r-project.org/web/packages/randomForestExplainer/randomForestExplainer.pdf))


#### Supervised Dimensionality Reduction
* [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
* [Discriminant analysis of principal components (DAPC)](https://grunwaldlab.github.io/Population_Genetics_in_R/DAPC.html)([R package - adegenet](http://adegenet.r-forge.r-project.org))

### Unsupervised
--------
Use cases: Structure Discovery, grouping/labeling when no labels are known, implicit recomendation, improve supervised methods
Examples: Youtube Recomendations, google translate 

#### Dimensionality Reduction
* PCA - [Practical Guide to Principal Component Analysis (PCA) in R & Python](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/)
* T-SNE - [Comprehensive Guide on t-SNE algorithm with implementation in R & Python](https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/) ([R Package](https://cran.r-project.org/web/packages/Rtsne/Rtsne.pdf))
* Vector Embeddings/Distributed Representations - [An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/) ([Python Package- gensim](http://gensim.readthedocs.io/en/latest/))
* Matrix Factorization - ([Python-Pytorch package - Spotlight](https://github.com/maciejkula/spotlight))
* Recommendation Systems - ([Introduction to Recommender System. Part 1 (Collaborative Filtering, Singular Value Decomposition)](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75))
* Autoencoders

#### Non-Parametric Classification
* Mixture Models - [mixtools - R Package](http://r.adu.org.za/web/packages/mixtools/vignettes/mixtools.pdf)
* Dirichlet Process Models [Python package](http://scikit-learn.org/stable/modules/mixture.html#the-dirichlet-process)
* [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) ([R package - dbscan](https://cran.r-project.org/web/packages/dbscan/dbscan.pdf))
--------
### Other Domains that are Interestingly Different

Use cases: A/B/C... tests, maintain equalibrium or setpoint, explore/exploit optimization 
Examples:Website optimization, content personalization, auto pilot

#### Autonomus Control/Decision Theory
* PID Algorithm - [PID controller](https://en.wikipedia.org/wiki/PID_controller) ([Python Code](https://github.com/ivmech/ivPID))
* Bandit Algorithms - [Introduction to Multi-Armed Bandits](http://slivkins.com/work/MAB-book.pdf) ([Python Code](https://github.com/johnmyleswhite/BanditsBook))

#### Reinforcement Learning 
* [Introduction to Various Reinforcement Learning Algorithms. Part I (Q-Learning, SARSA, DQN, DDPG)](https://medium.com/@huangkh19951228/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)

* [Introduction to Various Reinforcement Learning Algorithms. Part II (TRPO, PPO)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-part-ii-trpo-ppo-87f2c5919bb9)
--------
# Ensembles 

[Kaggle Ensambling Guide](https://mlwave.com/kaggle-ensembling-guide/)
* Voting
* Averaging
* Stacked ensembling/Blending

Be careful when you ensamble, "Here there be dragons."
* [Competing in a data science contest without reading the data](http://blog.mrtz.org/2015/03/09/competition.html)
* [The reusable holdout: Preserving validity in adaptive data analysis](http://science.sciencemag.org/content/349/6248/636.full)
--------
# Evaluating Accuracy

* [How data scientists can convince doctors that AI works](https://towardsdatascience.com/how-data-scientists-can-convince-doctors-that-ai-works-c27121432ccd)
* [Do machines actually beat doctors? ROC curves and performance metrics](https://lukeoakdenrayner.wordpress.com/2017/12/06/do-machines-actually-beat-doctors-roc-curves-and-performance-metrics/)




