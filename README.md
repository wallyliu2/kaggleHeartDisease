# kaggleHeartDisease

[A learning modeling framework]

    1. Exploratory data analysis (EDA): the process of going through a dataset and finding out more about it
    2. Model training: create model(s) to learn to predict a target variable based on other variables
    3. Model evaluation: evaluating a model’s predictions using problem-specific evaluation metrics
    4. Model comparison: comparing several different models to find the best one
    5. Model fine-tuning: once we’ve found a good model, how can we improve it?
    6. Feature Importance: since we’ve predicting the presence of heart disease, are there some things which are more important for prediction?
    7. Cross-validation: if we do build a good model, can we be sure it will work on unseen data?
    8. Reporting what we’ve found: if we had to present our work, what would we show someone?

[Steps]

Step 1. Define the problem
Step 2. Data
Step 3. Evaluation
Step 4. Features


[Terminologies]
* Hyperparameter tuning - Each model you use has a series of dials you can turn to dictate how they perform. Changing these values may increase or decrease model performance.
* Feature importance - If there are a large amount of features we're using to make predictions, do some have more importance than others? For example, for predicting heart disease, which is more important, sex or age?
* Confusion matrix - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).
* Cross-validation - Splits your dataset into multiple parts and train and tests your model on each part and evaluates performance as an average.
* Precision - Proportion of true positives over total number of samples. Higher precision leads to less false positives.
* Recall - Proportion of true positives over total number of true positives and false negatives. Higher recall leads to less false negatives.
* F1 score - Combines precision and recall into one metric. 1 is best, 0 is worst.
* ROC Curve - Receiver Operating Characterisitc is a plot of true positive rate versus false positive rate.
* Area Under Curve (AUC) - The area underneath the ROC curve. A perfect model achieves a score of 1.0.
* Classification report - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
* Support - The number of samples each metric was calculated on.
* Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0.
* Macro avg - Short for macro average, the average precision, recall and F1 score between classes. Macro avg doesn’t class imbalance into effort, so if you do have class imbalances, pay attention to this metric.
* Weighted avg - Short for weighted average, the weighted average precision, recall and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favour the majority class (e.g. will give a high value when one class out performs another due to having more samples).


Due to this project is about classification problem, we found some classifier model for this project usage, including Logistic Regression, K Neighbor Classifier, and Random Forest Classifier.

[Summary]

Load Data/Packages
1. Load libraries: numpy, pandas, matplotlib, seaborn
2. Load models: Logistic Regression, K Neighbor Classifier, Random Forest Classifier
3. Load evaluators: train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, confusion_matrix, classification_report, precision_score, recall_score, f1_score, plot_roc_curve
4. Load data: heart-disease.csv

EDA
1. Review the data by using head(), shape, value_counts(normalize=True), plot(), info(), describe(), crosstab()
2. Review the data by plotting: hist(), bar(), scatter()
3. Review independent variables by using corr() and plotted by heatmap()


Model Training, Evaluation, and Comparison
1. Drop target column from the data table as y
2. Split data set to 80% train and 20% test set by using train_test_split()
3. Create model training function pipeline and show the accuracy score


Here's the game plan:
1. Tune model hyperparameters, see which performs best
2. Perform cross-validation
3. Plot ROC curves
4. Make a confusion matrix
5. Get precision, recall and F1-score metrics
6. Find the most important model features

Model Fine-tuning (Parameters)
1. You can tune by hand, randomizedSearchCV, or GridSearchCV

The difference between RandomizedSearchCV and GridSearchCV is where RandomizedSearchCV searches over a grid of hyperparameters performing n_iter combinations, GridSearchCV will test every single possible combination.
In short:
* RandomizedSearchCV - tries n_iter combinations of hyperparameters and saves the best.
* GridSearchCV - tries every single combination of hyperparameters and saves the best.

Advanced Model Evaluation 
1. ROC curve and AUC score - plot_roc_curve()
2. Confusion matrix - confusion_matrix()
3. Classification report - classification_report()
4. Precision - precision_score()
5. Recall - recall_score()
6. F1-score - f1_score()

Cross-validation

Feature Importance
1. Model
2. Model.coef_
