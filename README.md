# Overview-of-Machine-Learning

## Informal Taxonomy of ML Algorithms
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
* KNN
* Decision Tree

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
* Matrix Factorization
* Autoencoders

#### Non-Parametric Classification
* Gaussian Mixture Models
* Dirichlet Process Models
* [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) ([R package - dbscan](https://cran.r-project.org/web/packages/dbscan/dbscan.pdf))

### Other Domains that are Interestingly Different
--------
Use cases: A/B/C... tests, maintain equalibrium or setpoint, explore/exploit optimization 
Examples:Website optimization, content personalization, auto pilot

#### Autonomus Control/Decision Theory
* PID Algorithm - [PID controller](https://en.wikipedia.org/wiki/PID_controller) ([Python Code](https://github.com/ivmech/ivPID))
* Bandit Algorithms - [Introduction to Multi-Armed Bandits](http://slivkins.com/work/MAB-book.pdf) ([Python Code](https://github.com/johnmyleswhite/BanditsBook))
* Reinforcement Learning

# Ensembles 
--------
[Kaggle Ensambling Guide](https://mlwave.com/kaggle-ensembling-guide/)
* Voting
* Averaging
* Stacked ensembling/Blending

Be careful when you ensamble, "Here there be dragons."
[Competing in a data science contest without reading the data](http://blog.mrtz.org/2015/03/09/competition.html)
[The reusable holdout: Preserving validity in adaptive data analysis](http://science.sciencemag.org/content/349/6248/636.full)

# Evaluating Accuracy
--------
* [How data scientists can convince doctors that AI works](https://towardsdatascience.com/how-data-scientists-can-convince-doctors-that-ai-works-c27121432ccd)
* [Do machines actually beat doctors? ROC curves and performance metrics](https://lukeoakdenrayner.wordpress.com/2017/12/06/do-machines-actually-beat-doctors-roc-curves-and-performance-metrics/)




