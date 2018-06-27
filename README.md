# identify_political_persuasion_on_reddit
Using a collection of Reddit posts to classify political persuasion and evaluate the performance of classification models.

This project has three stages:

1) Pre-processing reddit posts and labels - Applied tokenization, lemmetization, stemming and parts-of-speech tagging to corpus.
   Check preproc.py

2) Feature extraction - extracted 173 features (30 calculated in extractFeatures.py, 143 are pre-computed features for the dataset using      the LIWC tool)
   Check extractFeatures.py
 
3) Classification - trained a svc with linear kernel, svc with radial basis kernel, randomforestclassifier, mlpclassifier and                  adaboostclassifer on the training set and evaluated the accuracy, precision and recall.
   Used scikit-learn selectkbest to select the best features for the classifiers.
   Ran kfold cross-validation on each of the classifiers to see if the evaluation was the same as found initially.
   Check classify.py

Code:
prepoc.py
extractFeatures.py
classify.py

Data:
Alt, Center, Left, Right - reddit posts for each type of political class

Processing Files:
StopWords
abbrev.english - abbreviations

Feature Extraction:
BristolNorms+GilhoolyLogie.csv
Ratings_Warriner_et_al.csv
Slang
LIWC features and ids for each class




