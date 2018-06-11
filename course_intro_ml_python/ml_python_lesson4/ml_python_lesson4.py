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
    dataset = dataset.batch(5).repeat(epochs)
    
    # return the next batch of data
    (the_features, the_labels) = dataset.make_one_shot_iterator().get_next()
    return (the_features, the_labels)


# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# adjust median_house_value column
housing = housing[housing.median_house_value < 500000]
housing.median_house_value /= 1000

# set up rooms_per_person and clip to 0...4 range
housing["rooms_per_person"] = housing.total_rooms / housing.population
housing.rooms_per_person = housing.rooms_per_person.apply(lambda x: min(x, 4))

# calculate one-hot vector of binned latitudes
latitude_range = zip(range(32, 44), range(33, 45))
for r in latitude_range:
    housing["latitude_%d_to_%d" % r] = housing["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

# set up feature column names and descriptors
feature_columns = ["median_income", "rooms_per_person"]
for r in latitude_range:
    feature_columns.append("latitude_%d_to_%d" % r)
feature_column_desc = set([tf.feature_column.numeric_column(f) for f in feature_columns])

# shuffle housing data
housing = housing.reindex(np.random.permutation(housing.index))

# partition into training, validation, and test sets
training = housing[1:12000]
validation = housing[12001:14500]
test = housing[14501:17000]

# set up the model
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.05
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
model = tf.estimator.LinearRegressor(
    feature_columns = feature_column_desc,
    optimizer = optimizer
)

# train the model
print ("Training ML model (500 steps)...")
_ = model.train(
    input_fn = lambda: input_fn(
        training[feature_columns], 
        training.median_house_value),
    steps = 500
)
  
# validate the model
print ("Validating ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(
        validation[feature_columns], 
        validation.median_house_value,
        epochs = 1)
)

# get prediction values
prediction_values = np.array([item['predictions'][0] for item in prediction_set])

# calculate rmse
print ("Calculating RMSE for validation...")
mse = metrics.mean_squared_error(prediction_values, validation.median_house_value)
rmse = math.sqrt(mse)
print ("RMSE: ", rmse)

# test the model
print ("Testing ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(
        test[feature_columns], 
        test.median_house_value,
        epochs = 1)
)

# get prediction values
prediction_values = np.array([item['predictions'][0] for item in prediction_set])

# calculate rmse
print ("Calculating RMSE for test...")
mse = metrics.mean_squared_error(prediction_values, test.median_house_value)
rmse = math.sqrt(mse)
print ("RMSE: ", rmse)

