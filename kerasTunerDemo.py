'''https://www.tensorflow.org/tutorials/keras/keras_tuner'''
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

(img_train, label_train),(img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
# is used to change the data type of the elements in the array. 
# For example, you can convert an array of integers to an array of floats using astype(float). 'as type'
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512    
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32) # the Int function is part of the keras_tuner library.
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])


    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    ''' 
    Cross entropy loss 
    is a metric used to measure how well a classification model in machine learning performs. 
    The loss (or error) is measured as a number between 0 and 1, with 0 being a perfect model. 
    The goal is generally to get your model as close to 0 as possible.'''

    return model

# instantiate the tuner and perform hypertuning 
   # The Keras Tuner has four tuners available 
   # - RandomSearch, Hyperband, BayesianOptimization, and Sklearn. 
#In this tutorial, you use the Hyperband tuner.
tuner = kt.Hyperband(model_builder,
                      objective='val_accuracy',
                      max_epochs=10,
                      factor=3,
                      directory='my_dir',
                      project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# train the model
  ## Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=5, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch)) # the %d is a placeholder used in string formatting to represent an integer value

#Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
hypermodel = tuner.hypermodel.build(best_hps)

# retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# To finish this tutorial, evaluate the hypermodel on the test data.
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]: ", eval_result)


