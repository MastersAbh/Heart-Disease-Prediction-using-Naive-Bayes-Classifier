# Heart-Disease-Prediction-using-Naive-Bayes-Classifier
Implementation of naive bayes classifier in detecting the presence of heart disease using the records of previous patients.

Naive bayes classifier implemented from scratch without the use of any standard library and evaluation on the dataset available from UCI. Comparison of this model is made with Gaussian Naive Bayes Classifier of sklearn library. I have used 5 random partitions of training and testing dataset to evaluate the implementation and results are given for each partition.

Dataset for evaluating Naïve Bayes classifier: Heart Disease Data Set, available from: https://archive.ics.uci.edu/ml/datasets/Heart+Disease



heartdisease.txt is the dataset where 14 different attributes are taken for every patient.
The following 14 attributes were taken: age (in years), sex (male or female), cp (chest pain type), trestbps (resting blood pressure in mm Hg on admission to the hospital), chol (serum cholesterol in mg/dl) , restecg (resting electrocardiographic results), thalach (maximum heart rate achived), exang (exercise induced angina), oldpeak (ST depression induced by exercise relative to rest ), slope(the slope of the peak exercise ST segment), ca(number of major vessels (0-3) colored by flourosopy ) , thal( normal, fixed defect, reversable defect ), num (the predicted attribute) .


As the calssifier is taking random partitions for 5 different times, random results would be predicted with average accuracy being around 50.00 %.
