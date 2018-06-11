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



# add a scatterplot to a set of plots
#     features:   pandas column array with input features
#     labels:     pandas column with output labels
#     plotnumber: the number of the plot to draw
def add_plot(slope, intercept, features, labels, plotnumber, title):

    # draw the regression line
    x0 = features.min()
    x1 = features.max()
    y0 = slope * x0 + intercept 
    y1 = slope * x1 + intercept
    p = plt.subplot(1, 3, plotnumber)
    p.set_ylim([0, 600])
    p.set_xlim([0, 16])
    p.plot([x0, x1], [y0, y1], c='r')

    # show a scatterplot of features and labels
    p.set_title(title)
    p.set_xlabel("median income")
    p.set_ylabel("median house value")
    p.scatter(features, labels)
    p.plot()



# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# keep only records with median house values < 500,000
housing = housing[housing.median_house_value < 500000]

# convert the house value range to thousands
housing.median_house_value /= 1000

# shuffle housing data
housing = housing.reindex(np.random.permutation(housing.index))

# partition into training, validation, and test sets
training = housing[1:12000]
validation = housing[12001:14500]
test = housing[14501:17000]

# plot the test set
plt.figure(figsize=(13, 8))
p = plt.subplot(1, 3, 1)
p.set_ylim([32, 43])
p.set_xlim([-126, -112])
p.set_title("Test Set")
p.set_xlabel("longitude")
p.set_ylabel("latitude")
plt.scatter(test.longitude, 
            test.latitude,
            cmap = "coolwarm",
            c = test.median_house_value / test.median_house_value.max())

# plot the validation set
p = plt.subplot(1, 3, 2)
p.set_ylim([32, 43])
p.set_xlim([-126, -112])
p.set_title("Validation Set")
p.set_xlabel("longitude")
p.set_ylabel("latitude")
plt.scatter(validation.longitude, 
            validation.latitude,
            cmap = "coolwarm",
            c = validation.median_house_value / validation.median_house_value.max())

# plot the training set
p = plt.subplot(1, 3, 3)
p.set_ylim([32, 43])
p.set_xlim([-126, -112])
p.set_title("Training Set")
p.set_xlabel("longitude")
p.set_ylabel("latitude")
plt.scatter(training.longitude, 
            training.latitude,
            cmap = "coolwarm",
            c = training.median_house_value / training.median_house_value.max())
plt.plot()
plt.show(block = False)

# set up the model
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.00002
)
model = tf.estimator.LinearRegressor(
    feature_columns = [tf.feature_column.numeric_column("median_income")],
    optimizer = optimizer
)

# train the model
print ("Training ML model (500 steps)...")
_ = model.train(
    input_fn = lambda: input_fn(
        training[ ["median_income"] ], 
        training.median_house_value),
    steps = 500
)

# print the slope and intercept
slope = model.get_variable_value('linear/linear_model/median_income/weights')[0]
intercept = model.get_variable_value('linear/linear_model/bias_weights')
print ("Slope:", slope)
print ("Intercept:", intercept)

# plot the training result
plt.figure(figsize=(13, 8))
add_plot(slope, intercept, training.median_income, training.median_house_value, 1, "Training Set")

# validate the model
print ("Validating ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(
        validation[ ["median_income"] ], 
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

# calculate rmse as percentage of range
range = validation.median_house_value.max() - validation.median_house_value.min()
print ("Which is ", 100 * rmse / range, "% of the label range")

# plot the validation result
add_plot(slope, intercept, validation.median_income, validation.median_house_value, 2, "Validation Set")

# test the model
print ("Testing ML model...")
prediction_set = model.predict(
    input_fn = lambda: input_fn(
        test[ ["median_income"] ], 
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

# calculate rmse as percentage of range
range = test.median_house_value.max() - test.median_house_value.min()
print ("Which is ", 100 * rmse / range, "% of the label range")

# plot the validation result
add_plot(slope, intercept, test.median_income, test.median_house_value, 3, "Test Set")

# show the finished plot
plt.show()
