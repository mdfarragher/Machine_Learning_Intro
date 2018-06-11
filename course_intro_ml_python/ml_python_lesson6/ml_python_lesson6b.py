import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
from matplotlib import pyplot as plt
import os

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.width = 512

# tensorflow input function
#     features: pandas column array with input features
#     labels:   pandas column with output labels
#     epochs:   number of training epochs
#     returns:  a tuple with the current features and labels to train from
def input_fn(features, labels, epochs = None):

    # convert pandas features into a dictionary of numpy arrays
    feature_dict = { key: np.array(value) for (key, value) in dict(features).items() }                                           
 
    # construct a dataset
    dataset = Dataset.from_tensor_slices( (feature_dict, labels) ) 
    dataset = dataset.batch(20).repeat(epochs)
    
    # return the next batch of data
    (the_features, the_labels) = dataset.make_one_shot_iterator().get_next()
    return (the_features, the_labels)


# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# add classification feature
housing["median_high_house_value"] = (housing.median_house_value > 265000).astype(float)

# set up feature column names and descriptors
feature_columns = [
    "latitude",
    "longitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"]
feature_column_desc = set([tf.feature_column.numeric_column(f) for f in feature_columns])

# shuffle housing data
housing = housing.reindex(np.random.permutation(housing.index))

# partition into training, validation, and test sets
training = housing[1:12000]
validation = housing[12001:14500]
test = housing[14501:17000]

# set up the model
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.000005
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
model = tf.estimator.LinearClassifier(
    feature_columns = feature_column_desc,
    optimizer = optimizer
)

# train the model
print ("Training ML model (500 steps)...")
_ = model.train(
    input_fn = lambda: input_fn(
        training[feature_columns], 
        training.median_high_house_value),
    steps = 500
)

# validate the model
print ("Validating ML model...")
probability_set = model.predict(
    input_fn = lambda: input_fn(
        validation[feature_columns], 
        validation.median_high_house_value,
        epochs = 1)
)

# get probability values
probability_values = np.array([item['probabilities'][0] for item in probability_set])

## plot a histogram of predictions
plt.figure(figsize=(13, 8))
#plt.title("Probabilities histogram")
#plt.xlabel("probability value")
#plt.ylabel("frequency")
#plt.hist(probability_values, bins = 50)
#plt.plot()
#plt.show()

# print validation logloss
log_loss = metrics.log_loss(validation.median_high_house_value, probability_values)
print("Validtion LogLoss:", log_loss)

# get prediction values using a threshold of 0.7
prediction_values = [1 if value < 0.7 else 0 for value in probability_values]

# get classification scores
confusion = metrics.confusion_matrix(validation.median_high_house_value, prediction_values)
tn, fp, fn, tp = confusion.ravel()
print("True Positives: ", tp)
print("True Negatives: ", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

# get tpr and fpr
tpr = 1.0 * tp / (tp + fn)
fpr = 1.0 * fp / (fp + tn)
print("TPR:", tpr)
print("FPR:", fpr)

# print various evaluation metrics
accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
precision = 1.0 * tp / (tp + fp)
recall = 1.0 * tp / (tp + fn)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# get the auc value
print ("Evaluating ML model...")
evaluation_set = model.evaluate(
    input_fn = lambda: input_fn(
        validation[feature_columns], 
        validation.median_high_house_value,
        epochs = 1)
)
print ("AUC:", evaluation_set["auc"])

# get the roc curve
fpr, tpr, thresholds = metrics.roc_curve(validation.median_high_house_value, 1 - probability_values)
plt.title("Area under ROC curve")
plt.xlabel("FPR (1 - specificity)")
plt.ylabel("TPR (sensitivity)")
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0, 1], [0, 1], label="Random classifier")

plt.show()


