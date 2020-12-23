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


## Parameter Sampler
I implemented the Random Parameter Sampling as it supports discrete and continuous hyperparameters, early termination policies, and delivers similar results at a faster rate than other Parameter Sampler's such as Grid Sampling.

## Early Stopping Policy
I implemented a Bandit Policy as the three factors provided with implementing this policy (evaluation interval, delay evaluation, and slack factor) allow us to run experiments without having to stress over unneccessary experiments that run for a long period of time only to come out unneeded due to the primary metric not existing within the specified slack factor when compared to the best performing run.

## AutoML
1) Import Data (TabularDatasetFactory)
2) Clean Data
3) Split Data (Train & Test)
4) Configuration of AutoML
5) Save Best Model

## Pipeline Comparison
The differences between the models exist in their configurations. However, they follow similar data processing steps which resulted in similar accuracies (HyperDrive Approach - 91% & AutoML Approach - 92%). The AutoML Approach was more accurate, but took longer time to come up with the best model.

## Future Work
I would implement further visualizing widgets to better assess which groups might be negatively affected by a model and compare this with multiple models.

## Clean Up
