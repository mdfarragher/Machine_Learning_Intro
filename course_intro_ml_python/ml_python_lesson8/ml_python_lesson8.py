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
    dataset = dataset.batch(10).repeat(epochs)
    
    # return the next batch of data
    (the_features, the_labels) = dataset.make_one_shot_iterator().get_next()
    return (the_features, the_labels)


# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# adjust median_house_value column
housing = housing[housing.median_house_value < 500000]
housing.median_house_value /= 1000

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
    learning_rate = 0.001
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
model = tf.estimator.DNNRegressor(
    feature_columns = feature_column_desc,
    hidden_units = [8],
    optimizer = optimizer
)

# train the model
print("Training model...")
training_errors = []
validation_errors = []
for period in range (0, 20):

    # train the model
    _ = model.train(
        input_fn = lambda: input_fn(
            training[feature_columns], 
            training.median_house_value),
        steps = 50
    )

    # calculate training error
    training_probability_set = model.predict(input_fn = lambda: input_fn(training[feature_columns], training.median_house_value, epochs = 1))
    training_predictions = np.array([item['predictions'][0] for item in training_probability_set])

    # calculate validation error
    validation_probability_set = model.predict(input_fn = lambda: input_fn(validation[feature_columns], validation.median_house_value, epochs = 1))
    validation_predictions = np.array([item['predictions'][0] for item in validation_probability_set])

    # Compute training and validation loss.
    training_rmse = math.sqrt(metrics.mean_squared_error(training_predictions, training.median_house_value))
    validation_rmse = math.sqrt(metrics.mean_squared_error(validation_predictions, validation.median_house_value))

    # print progress
    print("  Period %02d : %0.2f" % (period, training_rmse))
    # print("  Period %02d : %0.2f, %0.2f" % (period, training_rmse, validation_rmse))

    # store rmse values for later plotting
    training_errors.append(training_rmse)
    validation_errors.append(validation_rmse)

# print final errors
print ("Final RMSE (on training data):   %0.2f" % training_rmse)
# print ("Final RMSE (on validation data): %0.2f" % validation_rmse)

# plot a graph of errors over time
plt.title("Model Training")
plt.xlabel("period")
plt.ylabel("RMSE")
plt.tight_layout()
plt.plot(training_errors, label = "training")
# plt.plot(validation_errors, label = "validation")
plt.legend()

plt.show()
