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

# load the housing data
print ("Loading data...")
housing = pd.read_csv("california_housing.csv")

# keep only records with median house values < 500,000
housing = housing[housing.median_house_value < 500000]

# convert the house value range to thousands
housing.median_house_value /= 1000

# add rooms per person column
housing["rooms_per_person"] = housing.total_rooms / housing.population

# shuffle housing data
housing = housing.reindex(np.random.permutation(housing.index))

# calculate correlation matrix
correlation = housing.corr()

# print correlation matrix
# print (correlation)

# calculate one-hot vector of binned latitudes
latitude_range = zip(range(32, 44), range(33, 45))
for r in latitude_range:
    housing["latitude_%d_to_%d" % r] = housing["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  
# print housing frame
# print(housing)

# clip rooms_per_person to the 0...4 range
housing.rooms_per_person = housing.rooms_per_person.apply(lambda x: min(x, 4))

# show rooms_per_person histogram
housing.rooms_per_person.hist(bins = 50)
plt.show()
