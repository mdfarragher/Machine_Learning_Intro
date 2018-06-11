import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import os

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.width = 512

# tensorflow training input function
#     features:   pandas column array with input features
#     labels:     pandas column with output labels
#     batch_size: size of data batches
#     epochs:     number of training epochs
#     shuffle:    indicates if data should be shuffled
#     returns:    a tuple with the current features and labels to train from
def create_training_fn(features, labels, batch_size, num_epochs = None, shuffle = True):

    def _input_fn(num_epochs = None, shuffle = True):

        # randomize all the data 
        idx = np.random.permutation(features.index)

        # create dataset object
        raw_features = {"pixels": features.reindex(idx)}
        raw_targets = np.array(labels[idx])
        ds = Dataset.from_tensor_slices((raw_features,raw_targets)) 
        ds = ds.batch(batch_size).repeat(num_epochs)

        # shuffle data if requested    
        if shuffle:
            ds = ds.shuffle(10000)
    
        # return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


# tensorflow prediction input function
#     features:   pandas column array with input features
#     labels:     pandas column with output labels
#     batch_size: size of data batches
#     returns:    a tuple with the current features and labels to predict with
def create_predict_fn(features, labels, batch_size):

    def _input_fn(num_epochs = None, shuffle = True):

        # create dataset object
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)
        ds = Dataset.from_tensor_slices((raw_features,raw_targets)) 
        ds = ds.batch(batch_size)

        # return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn




# load the mnist dataset
print ("Loading data...")
digits = pd.read_csv("handwritten_digits_large.csv", header = None)

# partition into training and validation sets
pivot = int(digits[0].count() * 0.8)
training = digits.loc[1:pivot, :]
validation = digits.loc[pivot+1:, :]

# extract training labels and features
training_labels = training.loc[:, 0]
training_features = training.loc[:, 1:784]
training_features = training_features / 255

# extract validation labels and features
validation_labels = validation.loc[:, 0]
validation_features = validation.loc[:, 1:784]
validation_features = validation_features / 255

# grab a random digit
#index = np.random.choice(training.index)

# plot the digit
#plt.matshow(training_features.loc[index].values.reshape(28, 28), cmap = 'Greys')
#plt.title("Label: %i" % training_labels.loc[index])
#plt.show()

# set up feature column descriptor
feature_column_desc = set([tf.feature_column.numeric_column('pixels', shape=784)])

# create input functions
batch = 30
fn_predict_training = create_predict_fn(training_features, training_labels, batch)
fn_predict_validation = create_predict_fn(validation_features, validation_labels, batch)
fn_training = create_training_fn(training_features, training_labels, batch)

# set up the model
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.05
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
model = tf.estimator.DNNClassifier(
    feature_columns = feature_column_desc,
      n_classes = 10,
      hidden_units = [100, 100],
      optimizer = optimizer,
      config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
)

# train the model
print("Training model...")
training_errors = []
validation_errors = []
for period in range (0, 10):

    # train the model
    _ = model.train(
        input_fn = fn_training,
        steps = 10
    )

    # calculate training predictions
    training_probability_set = list(model.predict(input_fn = fn_predict_training))
    training_probabilities = np.array([item['probabilities'][0] for item in training_probability_set])
    training_class_ids = np.array([item['class_ids'][0] for item in training_probability_set])
    training_predictions = tf.keras.utils.to_categorical(training_class_ids, 10)

    # calculate validation predictions
    validation_probability_set = list(model.predict(input_fn = fn_predict_validation))
    validation_probabilities = np.array([item['probabilities'][0] for item in validation_probability_set])
    validation_class_ids = np.array([item['class_ids'][0] for item in validation_probability_set])
    validation_predictions = tf.keras.utils.to_categorical(validation_class_ids, 10)

    # compute training and validation loss
    training_logloss = metrics.log_loss(training_labels, training_predictions)
    validation_logloss = metrics.log_loss(validation_labels, validation_predictions)

    # print progress
    print("  Period %02d : %0.2f, %0.2f" % (period, training_logloss, validation_logloss))

    # store rmse values for later plotting
    training_errors.append(training_logloss)
    validation_errors.append(validation_logloss)

# print final errors
print ("Final LogLoss (on training data): %0.2f" % training_logloss)
print ("Final LogLoss (on validation data): %0.2f" % validation_logloss)

# calculate final predictions
final_predictions = model.predict(input_fn = fn_predict_validation)
final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
  
# print final accuracy  
accuracy = metrics.accuracy_score(validation_labels, final_predictions)
print ("Final accuracy (on validation data): %0.2f" % accuracy)

# plot a graph of errors over time
p = plt.subplot(1, 2, 1)
p.set_title("Model Training")
p.set_xlabel("period")
p.set_ylabel("LogLoss")
p.plot(training_errors, label = "training")
p.plot(validation_errors, label = "validation")
p.legend()
p.plot()

# calculate the confusion matrix
cm = metrics.confusion_matrix(validation_labels, final_predictions)
cm_normalized = cm.astype("float") / cm.sum(axis = 1)[:, np.newaxis]

# plot the confusion matrix
p = plt.subplot(1, 2, 2)
ax = sns.heatmap(cm_normalized, cmap="bone_r", ax = p)
ax.set_aspect(1)
p.set_title("Confusion matrix")
p.set_ylabel("True label")
p.set_xlabel("Predicted label")
p.plot()

plt.show()
