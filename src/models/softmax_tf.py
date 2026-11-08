"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    np.random.seed(2020)
    tf.set_random_seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training

    train_x = tf.constant(train_x, dtype=tf.float32, name="image_pixel")

    train_y = tf.constant(train_y, dtype=tf.float32, name="classifier_out")

    test_x = tf.constant(test_x, dtype=tf.float32, name="test_pixel")

    val_x = tf.constant(val_x, dtype=tf.float32, name="val_pixel")

    # [TODO 2.8] Create weights (W) using TF variables 

    w = tf.Variable(tf.random.normal(shape=(785, 10), stddev = 0.01), name="weight")

    # [TODO 2.9] Create a feed-forward operator 

    z = tf.matmul(train_x, w)

    z_test = tf.matmul(test_x, w)

    z_val = tf.matmul(val_x, w)

    y_hat = tf.nn.softmax(z, name="probs")

    y_hat_test = tf.nn.softmax(z_test, name="probs_test")

    y_hat_val = tf.nn.softmax(z_val)

    # [TODO 2.10] Write the cost function

    eps = 1e-12

    y_hat = tf.clip_by_value(y_hat, eps, 1.0)

    y_hat_val = tf.clip_by_value(y_hat_val, eps, 1.0)

    cost = -tf.reduce_mean(tf.reduce_sum(train_y * tf.math.log(y_hat), axis=1))

    val_cost = -tf.reduce_mean(tf.reduce_sum(val_y * tf.math.log(y_hat_val), axis=1))

    # Define hyper-parameters and train-related parameters
    num_epoch = 3347
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    
    train_op = optimizer.minimize(cost)

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    patience = 20          
    min_delta = 1e-6      
    best_val_loss = np.inf
    wait = 0
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            #tic = time.clock()
            # [TODO 2.8] Compute losses and update weights here
            train_loss = sess.run(cost) 
            val_loss = sess.run(val_cost) 
            # Update weights
            sess.run(train_op)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            #toc = time.clock()
            #print(toc-tic)
            # [TODO 2.11] Define your own stopping condition here 

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss      
                wait = 0                      
            else:
                wait += 1                     
                if wait >= patience:
                    print(f'Early stop at epoch {e}, best val loss = {best_val_loss:.4f}')
                    break
          
        y_hat_test = sess.run(y_hat_test)
        test(y_hat_test, test_y)
