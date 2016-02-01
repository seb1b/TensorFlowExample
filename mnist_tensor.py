import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf



#initialize weigts randomly
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initialize bias positive to avoid dead neurons
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#stride one and padding on same to keep the original size
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def deepnn(t_cycles):
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])



    x_image = tf.reshape(x, [-1,28,28,1])

    #First Conv layer, there are 32 filters with a dimension of 5x5x1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #Second Conv layer 64 filters this time
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Third Fully Connected Layer going from 7*7*64 -> 1024
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #add Dropout for fc layer
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])


    #calcualte Softmax
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #define cost function with cross-entropy
    #reduce sum sums up over all images from given mini-batch
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

    #set up for training wich set length of 1e-4 and using ADAM optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #tf.argmax gives you the entry with the highest number
    # so tf.equality can easily check if the guessed class is correct
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #correct_prediction is a vector consisting of booleans like [true, ture, false] casting it now to 1 and zeros
    # with tf.mean we get then the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(t_cycles):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            #evaluates accuracy on current batch to get some feed back during training
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        #apply training steps with a batch size of 100 and keep_prob is for adjusting the dropout rate
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # no dropout in testing
    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})



def softmax_classifier():
    x = tf.placeholder(tf.float32,[None,784])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    y_ = tf.placeholder(tf.float32,[None,10])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


    #0.01 specifies the learning rate
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        #take a batch of 100 samples
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


#softmax_classifier()
#does MNIST classification with convnet using given training cycles
deepnn(500)



