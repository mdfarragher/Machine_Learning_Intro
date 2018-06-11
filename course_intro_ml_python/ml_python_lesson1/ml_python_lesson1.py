import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

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

# keep only records with median house values < 500,000
housing = housing[housing.median_house_value < 500000]

# set up a few series
median_income = housing.median_income
total_rooms = housing.total_rooms
median_house_value = housing.median_house_value

# convert the house value range to thousands
median_house_value /= 1000

# set up features and labels
feature_columns_desc = [tf.feature_column.numeric_column("median_income")]
features = housing[ ["median_income"] ]
labels = median_house_value

# set up the model
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.00002
)
model = tf.estimator.LinearRegressor(
    feature_columns = feature_columns_desc,
    optimizer = optimizer
)

# train the model
print ("Training ML model (500 steps)...")
_ = model.train(
    input_fn = lambda: input_fn(features, labels),
    steps = 500
)

# print the slope and intercept
slope = model.get_variable_value('linear/linear_model/median_income/weights')[0]
intercept = model.get_variable_value('linear/linear_model/bias_weights')
print ("Slope:", slope)
print ("Intercept:", intercept)


# validate the model
print ("Validating ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(features, labels, epochs = 1)
)

# get prediction values
prediction_values = np.array([item['predictions'][0] for item in prediction_set])

# calculate rmse
print ("Calculating RMSE...")
mse = metrics.mean_squared_error(prediction_values, labels)
rmse = math.sqrt(mse)
print ("RMSE: ", rmse)

# calculate rmse as percentage of range
range = labels.max() - labels.min()
print ("Which is ", 100 * rmse / range, "% of the label range")

# draw the regression line
x0 = median_income.min()
x1 = median_income.max()
y0 = slope * x0 + intercept 
y1 = slope * x1 + intercept
plt.plot([x0, x1], [y0, y1], c='r')

# show a scatterplot of features and labels
plt.xlabel("median income")
plt.ylabel("median house value")
plt.scatter(median_income, median_house_value)
plt.show()

