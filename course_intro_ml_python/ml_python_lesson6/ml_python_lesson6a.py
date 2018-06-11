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
    learning_rate = 0.000001
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
model = tf.estimator.LinearRegressor(
    feature_columns = feature_column_desc,
    optimizer = optimizer
)

# train the model
print ("Training ML model (200 steps)...")
_ = model.train(
    input_fn = lambda: input_fn(
        training[feature_columns], 
        training.median_high_house_value),
    steps = 200
)

# print model weights
weight_values = [model.get_variable_value("linear/linear_model/%s/weights" % name)[0][0] 
                 for name in feature_columns]
print(weight_values)

# validate the model
print ("Validating ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(
        validation[feature_columns], 
        validation.median_high_house_value,
        epochs = 1)
)

# get prediction values
prediction_values = np.array([item['predictions'][0] for item in prediction_set])

# plot a histogram of predictions
plt.figure(figsize=(13, 8))
plt.title("Predictions histogram")
plt.xlabel("prediction value")
plt.ylabel("frequency")
plt.hist(prediction_values, bins = 50)
plt.plot()

plt.show()
