# EEG-data-classification-using-Genetic-algorithms
In this repository, features were extracted from EEG data, and the best features were selected for classifying the signals using Genetic Algorithms.
## phase 1: 
- the features were extracted from the channels. 
- using the fisher criterion, the best features were chosen. 
- using 5-fold cross-validation, an MLP network was implemented to classify the data.
  -- * for better classification, the hyperparameters such as the number of layers, number of neurons, and activation functions were found. 
- the above step was done for implementing an RBF network whose hyperparameters, such as the number of neurons and their spreads, were optimized.
## phase 2:
- Feature selection was made using a Genetic Algorithm
- The precision of an SVM classifier was used as the objective function
- MLP and RBF neural networks were trained using selected features.
- Better outcomes were observed.
