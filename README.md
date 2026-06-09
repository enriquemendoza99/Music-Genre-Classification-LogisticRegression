# The Outlier Detectives - Project 2: Logistic Regression and Friends
## Instructions
1. Create a virtual environment using 'python3 -m venv runenv'
2. Activate the virtual environment using 'source runenv/bin/activate'
3. Install packages listed in 'requirements.txt' using pip in the virtual enviornment
4. Specify path to the training data in lines 54-63
5. Specify path to the Kaggle test data on line 628
6. Uncomment or comment out necessary lines of code for wanted tests, evaluation metrics, or hyperparameter tuning experiments
7. Run code in virtual environment using 'python3 dataProcessing.py'

## File Manifest
1. 'dataProcessing.py' - This is the main executable. This includes all the code used to pre-process the data, train the logisitic regression model, predict results based on the trained model, perform evaluation metrics on all ML models, and run hyperparameter optimization.
2. 'LogisticRegression.py' - This is where the logistic regression model is built.
3. 'README.md' - Includes instructions on how to run 'dataProcessing.py' and specify paths to necessary data.
4. 'requirements.txt' - Includes all necessary packages and versions that need to be installed to run 'dataProcessing.py' successfully

## Contributions
1. John Dominguez-Trujillo: Wrote the code in 'dataProcessing.py' that read the data, pre-processed the data (normalization, scaling, PCA), trained the logisitc regression model on the training dataset, predicted on the validation data set, computed evaluation metrics for the validation predictions, performed by-hand trial and error of hyperparameters to receive the best accuracy, implemented the Kaggle test data, pre-processed the Kaggle test data, predicted for the Kaggle test, and wrote Kaggle predictions to CSV. Added the 'LogisticRegressionSimplified' in 'LogisticRegression.py'. Helped write the PDF report.
2. Enrique Mendoza: Wrote the code in 'LogisticRegression.py' that built the logisitic regression model, implemented weight and bias into logistic regression, implemented gradient descent into logistic regression. Wrote the code for SVM Analysis and GBC Analysis in 'dataProcessing.py'. Helped write the PDF Report.
3. Alexa Martinez: Wrote the code for GNB Analysis and Random Forest Analysis in 'dataProcessing.py', implemented the hyperparameter tuning experiments and metrics for logisitic regression, support vector machine, gradient boosting classifier, and random forest classifier in 'dataProcessing.py'. Setup and helped write the PDF Report.

## Kaggle Outcome
1. Kaggle Date Run: November 8, 2025
2. Kaggle Accuracy Score: 0.62000 -> 62.000%
