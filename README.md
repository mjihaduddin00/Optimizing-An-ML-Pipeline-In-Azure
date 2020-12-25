# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This project utilizes UCI Bank Marketing information as our dataset which directly correlates with marketing campaigns of an institution. The goal is to predict if the client will subscribe to a term deposit. Thus, to gather a prediction we utilized Scikit-Learn Logistic Regression, tuned the hyperparameters (using HyperDrive), implement AutoML to optimize a model (with the same dataset to compare the results between the two methods). The conclusion was that the best performing model was obtained from AutoML with the algorithm being a VotingEnsemble with an accuracy of around 92%.

## Scikit-Learn Pipeline
1) Setup Train.py Script:
- Import Data (TabularDatasetFactory)
- Clean Data
- Split Data (Train & Test)
- Implement Logistic Regression Model

2) Scikit-Learn Estimator transfered for and to hyperdrive configuration

3) Within HyperDrive: 
- Select the parameter sampler
- Select Primary Metric
- Select Early Termination Policy
- Select Estimator
- Allocate Resources
- Additional Configuration

4) Save Trained Optimized Model

Having implemented Logistic Regressional for our model in this binary classification problem with the aid of the HyperDrive tool to choose the best hyperparameter values from the parameter search space we can view the logistic (sigmoidal) function at work that leads to the estimations between the dependent/target variable and one or more independent variables. In the screenshots section, it can be viewed which HyperDrive Run gave the best results. For further clarification, the parameters being used in this model are the following:

- Inverse of Regularization Strength (parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization"))
- Maximum Number of Iterations to Converge (parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge"))


## Parameter Sampler
I implemented the Random Parameter Sampling as it supports discrete and continuous hyperparameters, early termination policies, and delivers similar results at a faster rate than other Parameter Sampler's such as Grid Sampling. The reason for this is because it does not perform an exhaustive search making it less costly during experimentation as well.

## Early Stopping Policy
I implemented a Bandit Policy as the three factors provided with implementing this policy (evaluation interval, delay evaluation, and slack factor) allow us to run experiments without having to stress over unneccessary experiments that run for a long period of time only to come out unneeded due to the primary metric not existing within the specified slack factor when compared to the best performing run. This makes the Bandit Policy more superior than having no policy whatsoever as this would allow the experiment to run unnecessarily allowing for a larger run time as well. A Truncation Selection Policy was not implemented since we were unaware of how the model was going to perform meaning that at certain intervals we cannot terminate a run since we were unsure of what percentage the model was going to perform. We also did not implement a Media Stopping Policy as we were not looking at averages.

## AutoML
1) Import Data (TabularDatasetFactory)
2) Clean Data
3) Split Data (Train & Test)
4) Configuration of AutoML
5) Save Best Model

This led to the discovery that the VotingEnsemble was the best model for our dataset in terms of prediction and accuracy as it highlights feature important values, pattern discovery in data during training, and different metrics with their different values for model interpretability and explanation. This can be viewed further in the screenshots folder. The VotingEnsemble in this Classification Model is predicting the class with the largest summed probability from models which means that it utilizes features such as K-Nearest Neighbors (in Explanations (preview) we see that as Global Importance of Emp.Var.Rate being greater than 40% and in Summary Importance as well as the Confusion Matrix displaying a 96% True Label at 0, 0 and a 43% true label at 0, 1) as parameters to make its determination labeling this model to be a Soft Voting - VotingEnsemble Classification Model.

## Pipeline Comparison
The differences between the models exist in their configurations. However, they follow similar data processing steps which resulted in similar accuracies (HyperDrive Approach - 91% & AutoML Approach - 92%). The AutoML Approach was more accurate, but took longer time to come up with the best model. The ML Model is fixed in our approach without AutoML and HyperDrive is utilized to find optimal hyperparameters whereas with the AutoML approach different models are automatically generated with their own optimal hyperparameter values allowing for the best model to be selected.

## Future Work
I would utilize additional interactive visualizations to determine which groups of users are being negatively impacted by a model. I would then compare multiple models in terms of their fairness and performance with these impacted groups in mind. I would also look into extending the parameter search space to see the effect it has on determining the model and accuracy. I would also have liked to export the model via ONNX to be able to have the experiment available via my mobile device.

## Clean Up
The cluster was deleted as can be viewed in the screenshots folder.
