import pickle
import tarfile
import tensorflow as tf
import os
import sys
import numpy as np

CIFAR_FILENAME = "cifar-10-python.tar.gz"
CIFAR_DOWNLOAD_URL = "http://www.cs.toronto.edu/~kriz/" + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = "cifar-10-batches-py"
DATA_DIR = "./data/"
NUM_CLASSES = 10

tf.logging.set_verbosity(tf.logging.INFO)


def unpickle(file):    
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def load():
    training_set = range(1, 6)
    train_data = np.zeros((0, 3072), dtype=np.float16)
    train_labels = np.zeros((0,), dtype=np.int32,)
    test_data = np.zeros((0, 3072), dtype=np.float16)
    test_labels = np.zeros((0,), dtype=np.int32,)

    for num in training_set: 
        name = "{}/{}/data_batch_{}".format(DATA_DIR, CIFAR_LOCAL_FOLDER, num)
        data = unpickle(name)
        
        train_data = np.append(train_data, data[b"data"], axis=0)
        train_labels = np.append(train_labels, data[b"labels"], axis=0)
    
    name = "{}/{}/test_batch".format(DATA_DIR, CIFAR_LOCAL_FOLDER)
    data = unpickle(name)

    test_data = np.append(test_data, data[b"data"], axis=0)
    test_labels = np.append(test_labels, data[b"labels"], axis=0)

    train_data /= 255
    test_data /= 255

    #train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    return (train_data, train_labels, test_data, test_labels)
        

def model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
    
    conv1 = tf.layers.conv2d(
      inputs=input_layer, filters=64, kernel_size=[5, 5], padding='same',
      activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
    norm1 = tf.nn.lrn(
        pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # 2nd Convolutional Layer                                                                                                                 
    conv2 = tf.layers.conv2d(
        inputs=norm1, filters=64, kernel_size=[5, 5], padding='same',
        activation=tf.nn.relu, name='conv2')
    norm2 = tf.nn.lrn(
        conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.layers.max_pooling2d(
        inputs=norm2, pool_size=[3, 3], strides=2, name='pool2')

    # Flatten Layer                                                                                                                           
    shape = pool2.get_shape()
    pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])

    # 1st Fully Connected Layer                                                                                                               
    dense1 = tf.layers.dense(
        inputs=pool2_, units=384, activation=tf.nn.relu, name='dense1')

    # 2nd Fully Connected Layer                                                                                                               
    dense2 = tf.layers.dense(
        inputs=dense1, units=192, activation=tf.nn.relu, name='dense2')

    # 3rd Fully Connected Layer (Logits)                                                                                                      
    logits = tf.layers.dense(
        inputs=dense2, units=10, activation=tf.nn.relu, name='logits')
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_data, train_labels, test_data, test_labels = load()

    # Create the Estimator
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="/tmp/cifar_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    
    cifar10_classifier.train(
      input_fn=train_input_fn,
      steps=100,
      hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
    
    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results) 

if __name__ == "__main__":
    tf.app.run()
