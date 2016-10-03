import tensorflow as tf
import input_data

def main(parms):

  # Load the MNIST dataset
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, splits = parms['splits'])

  n_data = parms['n_data']
  n_data_feat = parms['n_data_feat']
  n_labels = parms['n_labels']
  assert(mnist.train.images.shape[0] == n_data)
  assert(mnist.train.images.shape[1] == n_data_feat)
  assert(mnist.train.labels.shape[0] == n_data)
  assert(mnist.train.labels.shape[1] == n_labels)

  # define the input and target placeholders
  data_p = tf.placeholder(tf.float32, shape = [None, n_data_feat])
  target_p = tf.placeholder(tf.float32, shape = [None, n_labels])

  if (parms['model_type'] == 'logreg'):
    # -- model #1 --
    # define the model parameters
    W = tf.Variable(tf.truncated_normal([n_data_feat, n_labels]))
    b = tf.Variable(tf.zeros([n_labels]))
    # define the model architecture
    logits = tf.matmul(data_p, W) + b
  elif (parms['model_type'] == 'fc'):
    # -- model #2 --
    fc1_W = tf.Variable(tf.truncated_normal([n_data_feat, 64], stddev = parms['init_variance']))
    fc1_b = tf.Variable(tf.zeros([64]))
    fc2_W = tf.Variable(tf.truncated_normal([64, n_labels], stddev = parms['init_variance']))
    fc2_b = tf.Variable(tf.zeros([n_labels]))
    # model arch
    h_fc1 = tf.nn.relu(tf.matmul(data_p, fc1_W) + fc1_b)
    logits = tf.matmul(h_fc1, fc2_W) + fc2_b
  elif (parms['model_type'] == 'conv'):
    # -- model #3 --
    conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = parms['init_variance']))
    conv1_b = tf.Variable(tf.zeros([32]))
    conv2_w = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = parms['init_variance']))
    conv2_b = tf.Variable(tf.zeros([64]))
    fc1_W = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev = parms['init_variance']))
    fc1_b = tf.Variable(tf.zeros([1024]))
    fc2_W = tf.Variable(tf.truncated_normal([1024, n_labels], stddev = parms['init_variance']))
    fc2_b = tf.Variable(tf.zeros([n_labels]))
    # model arch
    image = tf.reshape(data_p, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(image, conv1_w, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_b)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv2_w, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_b)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7*7*64]), fc1_W) + fc1_b)
    logits = tf.matmul(h_fc1, fc2_W) + fc2_b

  # evaluate loss from net output and target
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, target_p))
  # prediction is softmax of logigs
  prediction = tf.nn.softmax(logits)
  # define the evaluation measure
  correct_prediction = tf.equal(tf.argmax(target_p, 1), tf.argmax(prediction, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # define the optimizer
  if (parms['optimizer_type'] == 'graddesc'):
    optimizer = tf.train.GradientDescentOptimizer(parms['lambda'])
  elif (parms['optimizer_type'] == 'momentum'):
    optimizer = tf.train.MomentumOptimizer(parms['lambda'], momentum = 0.9)
  elif (parms['optimizer_type'] == 'ADAM'):
    optimizer = tf.train.AdamOptimizer(parms['lambda'])
  # TODO: try decaying learning rate (see TF API: tf.train.exponential_decay)
  optimization_step = optimizer.minimize(loss)

  saver = tf.train.Saver()

  sess = tf.Session()

  sess.run(tf.initialize_all_variables())

  cur_epoch = 0
  best_val_accuracy = 0
  lookahead_counter = 0
  # start training
  while (True):
    # get the next batch
    batch_data, batch_targets = mnist.train.next_batch(parms['batch_size'])
    # evaluate net outputs, loss and modify net params
    feed_dict = {data_p: batch_data, target_p: batch_targets}
    _, batch_loss, batch_predictions = sess.run([optimization_step, loss, prediction], feed_dict = feed_dict)
    # if we have finished a pass through training data (an epoch)
    if (mnist.train.epochs_completed != cur_epoch):
      if (parms['low_memory']):
        val_accuracy = 0.0
        while(mnist.validation.epochs_completed == cur_epoch):
          batch_val_data, batch_val_targets = mnist.validation.next_batch(parms['batch_size'])
          val_accuracy += sum(sess.run(correct_prediction, feed_dict = {data_p: batch_val_data, target_p: batch_val_targets}))
        val_accuracy /= mnist.validation.num_examples
        print('Epoch #%d: validation_accuracy = %f' % (cur_epoch, val_accuracy))
      else:
        [val_loss, val_accuracy] = sess.run([loss, accuracy], feed_dict = {data_p: mnist.validation.images, target_p: mnist.validation.labels})
        print('Epoch #%d: validation loss = %f, validation_accuracy = %f' % (cur_epoch, val_loss, val_accuracy))
      if (val_accuracy >= best_val_accuracy):
        print('Found best validation accuracy, saving model')
        best_val_accuracy = val_accuracy
        saver.save(sess, parms['checkpoint_path'])
        lookahead_counter = 0
      else:
        lookahead_counter += 1
      cur_epoch = mnist.train.epochs_completed
    if (mnist.train.epochs_completed == parms['max_epochs'] or
        lookahead_counter == parms['max_lookahead']):
      break

  # load the best model
  saver.restore(sess, parms['checkpoint_path'])
  # evaluate performance on test set
  if (parms['low_memory']):
    test_accuracy = 0.0
    while(not mnist.test.epochs_completed):
      batch_test_data, batch_test_targets = mnist.test.next_batch(parms['batch_size'])
      test_accuracy += sum(sess.run(correct_prediction, feed_dict = {data_p: batch_test_data, target_p: batch_test_targets}))
    test_accuracy /= mnist.test.num_examples
  else:
    test_accuracy = sess.run(accuracy, feed_dict = {data_p: mnist.test.images, target_p: mnist.test.labels})
  print('Test accuracy: %f' % (test_accuracy))

if __name__ == '__main__':

  splits = [55000, 5000, 5000] # [n_train, n_val, n_test]

  parms = { 'splits' : splits,
            'n_data' : splits[0],
            'n_data_feat' : 784,
            'n_labels' : 10,
            'model_type' : 'conv',
            'lambda' : 1e-4, # 0.5,
            'init_variance': 0.1,
            'optimizer_type' : 'ADAM', # 'graddesc'
            'max_epochs' : 200, # 30,
            'max_lookahead' : 5,
            'batch_size' : 50,
            'checkpoint_path' : '/tmp/model.ckpt',
            'low_memory': False }

  main(parms)
