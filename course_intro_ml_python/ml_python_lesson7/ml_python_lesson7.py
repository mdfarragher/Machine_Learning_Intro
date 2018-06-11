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
    dataset = dataset.batch(100).repeat(epochs)
    
    # return the next batch of data
    (the_features, the_labels) = dataset.make_one_shot_iterator().get_next()
    return (the_features, the_labels)


# get binned feature buckets
#     feature: pandas column array with input features
#     buckets: the number of buckets
#     returns:  a list of bucket boundaries
def get_buckets(feature, buckets):
  boundaries = np.arange(1.0, buckets) / buckets
  quantiles = feature.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]


# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# add classification feature
housing["median_high_house_value"] = (housing.median_house_value > 265000).astype(float)

# set up rooms_per_person and clip to 0...4 range
housing["rooms_per_person"] = housing.total_rooms / housing.population
housing.rooms_per_person = housing.rooms_per_person.apply(lambda x: min(x, 4))

# set up numeric column descriptors
longitude = tf.feature_column.numeric_column("longitude")
latitude = tf.feature_column.numeric_column("latitude") 
median_income = tf.feature_column.numeric_column("median_income")
rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
# bin longitude into 10 buckets.
binned_longitude = tf.feature_column.bucketized_column(
    longitude, 
    boundaries = get_buckets(housing.longitude, 10))
  
# bin latitude into 10 buckets.
binned_latitude = tf.feature_column.bucketized_column(
    latitude, 
    boundaries = get_buckets(housing.latitude, 10))

# add a feature cross of longitude x latitude
long_x_lat = tf.feature_column.crossed_column(
    set([binned_longitude, binned_latitude]), hash_bucket_size=1000) 

# set up feature column descriptors
feature_column_desc = [
    binned_longitude,
    binned_latitude,
    median_income,
    rooms_per_person,
    long_x_lat]

# set up feature column names
feature_columns = ["median_income", "rooms_per_person", "longitude", "latitude"]

# shuffle housing data
housing = housing.reindex(np.random.permutation(housing.index))

# partition into training, validation, and test sets
training = housing[1:12000]
validation = housing[12001:14500]
test = housing[14501:17000]

# set up the model
optimizer = tf.train.FtrlOptimizer(
    learning_rate = 0.1,
    l1_regularization_strength = 50
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
  
# print model weights
weight_values = [value for arr in model.get_variable_value("linear/linear_model/latitude_bucketized_X_longitude_bucketized/weights") for value in arr]
print("Nonzero weights:", np.count_nonzero(weight_values))

# plot a histogram of weights
plt.figure(figsize=(13, 8))
plt.title("Weights histogram")
plt.xlabel("weight value")
plt.ylabel("frequency")
plt.ylim(0, 50)
plt.hist(weight_values, bins = 50)
plt.plot()

plt.show()
