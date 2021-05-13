# Note: for some reason, there are going to be datapoints that fall outside the ideal prediction line no matter how many epochs are ran. Thus, I settled on a number
# of 250 as I thought that anything much higher would result in too long of a runtime and anything lower would be too inaccurate. I think that since the 
# expenses column has such a great range of values for sets of data that had similar characteristics, it was harder to predict some cases accurately. Thus,
# most of the results fall within a prediction error of 0 but a few have errors in the thousands. 


# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')


dataset = dataset.replace(['male','female'],
                [0,1])

dataset['sex'] = dataset['sex'].map({0: 'male', 1: 'female'})
dataset = pd.get_dummies(dataset, columns=['sex'], prefix='', prefix_sep='')


dataset = dataset.replace(['no','yes'],
                          [0,1])


dataset['smoker'] = dataset['smoker'].map({0: 'no', 1: 'yes'})
dataset = pd.get_dummies(dataset, columns=['smoker'], prefix='', prefix_sep='')


dataset = dataset.replace(['northeast','northwest','southeast','southwest'],
                          [0,1,2,3])

dataset['region'] = dataset['region'].map({0: 'northeast', 1: 'northwest', 2: 'southeast', 3: 'southwest'})
dataset = pd.get_dummies(dataset, columns=['region'], prefix='', prefix_sep='')

# Training and testing values
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')


# Normalization
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_dataset))


# Building model 
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


dnn_model = build_and_compile_model(normalizer)
history = dnn_model.fit(
    train_dataset, train_labels,
    validation_split=0.2,
    verbose=0, epochs=250)
test_predictions = dnn_model.predict(test_dataset).flatten()


# Graphing data / results 
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 100000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
