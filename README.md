# Rock_and_Mine_detector_with_KNN
Detecting a rock or mine using KNN

Sonar (sound navigation ranging) is a technique that uses sound propagation (usually underwater, as in submarine navigation) to navigate, communicate with or detect objects on or under the surface of the water, such as other vessels.

The data set contains the response metrics for 60 separate sonar frequencies sent out against a known mine field (and known rocks). These frequencies are then labeled with the known object they were beaming the sound at (either a rock or a mine).

Our main goal is to create a machine learning model capable of detecting the difference between a rock or a mine based on the response of the 60 separate sonar frequencies.

The provided Python code performs a machine learning task using a dataset called "sonar.all-data.csv." Here is a detailed description of the code:

**Importing Required Libraries:**

The code begins by importing the necessary libraries:
numpy (as np) for numerical operations.
pandas (as pd) for data manipulation and analysis.
seaborn for data visualization.
matplotlib.pyplot (as plt) for creating plots.
Loading the Dataset:

It reads the dataset "sonar.all-data.csv" into a pandas DataFrame named df.
Checking the Dataset's Shape:

The code prints the shape of the dataset, which is (208, 61), indicating there are 208 rows and 61 columns.
Displaying the First Few Rows of the Dataset:

df.head() is used to display the first few rows of the dataset, giving a glimpse of the data.
Data Exploration:

The code sets the stage for data exploration and visualization.
It separates the features (X) from the target variable (y), where 'Label' is the target variable.
It also maps the labels 'R' and 'M' to numerical values 0 and 1, respectively, to facilitate correlation analysis.
Calculating Correlations:

The code calculates the absolute correlation values between the features and the target variable ('Target').
The top 5 correlated frequencies with the target variable are displayed.
Creating a Correlation Heatmap:

A heatmap of the correlation matrix between all features is plotted using seaborn. This helps visualize the relationships between different frequency responses.
Train-Test Split:

The dataset is split into training and testing sets using train_test_split from scikit-learn. 90% of the data is used for training, and 10% for testing. The random_state is set to 42 for reproducibility.
Creating a Pipeline:

A machine learning pipeline is created using scikit-learn's Pipeline class.
The pipeline includes two steps: standard scaling (StandardScaler) and a k-nearest neighbors classifier (KNeighborsClassifier).
Grid Search for Hyperparameter Tuning:

A grid search is performed using GridSearchCV to find the best value of the hyperparameter 'k' for the k-nearest neighbors classifier. It tests various values of 'k' and uses 5-fold cross-validation to evaluate performance.
Displaying Best Parameters:

The best hyperparameters found by the grid search are displayed, including the optimal 'k' value.
Plotting Mean Test Scores vs. 'k':

The code creates a plot showing the mean test scores for different values of 'k' from the grid search results. This helps visualize how the classifier's performance varies with 'k'.
Final Model Evaluation:

The final step involves evaluating the model's performance on the test set.
Predictions are made on the test set using the best estimator found during grid search.
A confusion matrix and a classification report, including precision, recall, and F1-score, are generated to assess the model's performance.
Overall, this code demonstrates a machine learning workflow that includes data exploration, preprocessing, hyperparameter tuning, and model evaluation using a k-nearest neighbors classifier.
