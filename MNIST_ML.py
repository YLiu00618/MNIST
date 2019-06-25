"""
Explore the behaviors of five supervised learning techniques on the MNIST dataset and a simple toy dataset.
1). decision trees (DT),
2). neural networks (NN),
3). boosted trees (BST-DT),
4). support vector machines (SVM) and
5). k-nearest neighbors (kNN).

"""


from absl import app
from absl import flags
import numpy as np
import sklearn.datasets
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
import tensorflow as tf
import time


FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset', 'toy', ['toy', 'mnist'], 'Choose dataset name')

flags.DEFINE_enum('method', 'tree', ['tree', 'boosting', 'knn', 'svm', 'nn'],
    'Choose a method')

flags.DEFINE_integer('tree_max_depth', 10, 'Max depth for the tree.')

flags.DEFINE_enum('tree_criterion', 'gini', ['gini', 'entropy', 'random'],
                  'Tree branch split methods')

flags.DEFINE_integer('boosting_num_trees', 5, 'Number of trees for the boosting method.')

flags.DEFINE_integer('knn_k', 2, 'K in KNN')

flags.DEFINE_string('svm_kernel', 'linear', 'Kernel type')

flags.DEFINE_multi_integer('nn_depths', [], 'Depths of fully-connected layers')

flags.DEFINE_integer('nn_epochs', 5, 'Epochs to train for NN')

flags.DEFINE_float('train_ratio', 1, 'A percentage of training data is sampled for training')


def get_toy_dataset():
  """
  Toy dataset: a binary classification problem on the 2D plane.
  Toy dataset generation:
   1). choose 2 centers and assign to opposite labels;
   2). sample 1000 samples from an isotropical Gaussian distribution;
   3). find a concentral circle that split half of the samples from the other half;
   4). flip the labels for the samples outside of the circle.
  """

  n_samples = 2000
  n_half = n_samples / 2

  X1, y1 = sklearn.datasets.make_gaussian_quantiles(cov=2.,
                                   n_samples=n_half, n_features=2,
                                   n_classes=2, random_state=1)
  X2, y2 = sklearn.datasets.make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                   n_samples=n_half, n_features=2,
                                   n_classes=2, random_state=1)
  X = np.concatenate((X1, X2))
  y = np.concatenate((y1, - y2 + 1))

  p = np.random.permutation(X.shape[0])
  X = X[p]
  y = y[p]


  n_test = n_samples / 5

  train_X = X[:-n_test]
  train_y = y[:-n_test]

  test_X = X[-n_test:]
  test_y = y[-n_test:]

  num_classes = 2

  if FLAGS.train_ratio < 1.0:
    num_train = train_X.shape[0]
    train_X = train_X[:int(num_train * FLAGS.train_ratio)]
    train_y = train_y[:int(num_train * FLAGS.train_ratio)]

  return train_X, train_y, test_X, test_y, num_classes


def get_mnist():
  mnist = tf.keras.datasets.mnist
  (train_X, train_y),(test_X, test_y) = mnist.load_data()
  train_X, test_X = train_X / 255.0, test_X / 255.0
  num_classes = 10
  train_X = train_X.reshape((60000, 28 * 28))
  test_X = test_X.reshape((10000, 28 * 28))

  if FLAGS.train_ratio < 1.0:
    num_train = train_X.shape[0]
    train_X = train_X[:int(num_train * FLAGS.train_ratio)]
    train_y = train_y[:int(num_train * FLAGS.train_ratio)]

  return train_X, train_y, test_X, test_y, num_classes


def knn(train_X, train_y, test_X):
  start = time.time()
  print('\n')
  print('\n')
  print('-------------------------------------------------')
  print('Starting KNN training with k: %d ...' % FLAGS.knn_k)
  clf = sklearn.neighbors.KNeighborsClassifier(
      n_neighbors=FLAGS.knn_k, algorithm='kd_tree', n_jobs=4)
  clf.fit(train_X, train_y)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Eval ...')
  pred = clf.predict(test_X)
  print('... finished after %f secs' % (time.time() - start))
  return pred


def tree(train_X, train_y, test_X):
  start = time.time()
  print('\n')
  print('\n')
  print('-------------------------------------------------')
  print('Starting Tree training with max_depth: %d ...' % FLAGS.tree_max_depth)

  if FLAGS.tree_criterion == 'gini':
    criterion = 'gini'
    splitter = 'best'
  elif FLAGS.tree_criterion == 'entropy':
    criterion = 'entropy'
    splitter = 'best'
  elif FLAGS.tree_criterion == 'random':
    criterion = 'gini'
    splitter = 'random'

  clf = sklearn.tree.DecisionTreeClassifier(
      max_depth=FLAGS.tree_max_depth, random_state=0,
      criterion=criterion, splitter=splitter)
  clf.fit(train_X, train_y)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Eval ...')
  pred = clf.predict(test_X)
  print('... finished after %f secs' % (time.time() - start))
  return pred


def boosting(train_X, train_y, test_X):
  start = time.time()
  print('\n')
  print('\n')
  print('-------------------------------------------------')
  print('Starting boosting training with num_trees: %d, max_depth: %d ...' %
      (FLAGS.boosting_num_trees, FLAGS.tree_max_depth))

  if FLAGS.tree_criterion == 'gini':
    criterion = 'gini'
    splitter = 'best'
  elif FLAGS.tree_criterion == 'entropy':
    criterion = 'entropy'
    splitter = 'best'
  elif FLAGS.tree_criterion == 'random':
    criterion = 'gini'
    splitter = 'random'

  clf = sklearn.ensemble.AdaBoostClassifier(
      sklearn.tree.DecisionTreeClassifier(
          max_depth=FLAGS.tree_max_depth, criterion=criterion, splitter=splitter),
      algorithm='SAMME',
      n_estimators=FLAGS.boosting_num_trees)
  clf.fit(train_X, train_y)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Eval ...')
  pred = clf.predict(test_X)
  print('... finished after %f secs' % (time.time() - start))
  return pred


def svm(train_X, train_y, test_X):
  start = time.time()
  print('\n')
  print('\n')
  print('-------------------------------------------------')
  print('Starting SVM training with kernel: %s ...' % FLAGS.svm_kernel)

  if FLAGS.svm_kernel.startswith('poly'):
    degree = int(FLAGS.svm_kernel.split('-')[1])
    kernel = 'poly'
  else:
    kernel = FLAGS.svm_kernel
    degree = 1

  clf = sklearn.svm.SVC(kernel=kernel, degree=degree)
  clf.fit(train_X, train_y)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Train Eval ...')
  train_pred = clf.predict(train_X)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Eval ...')
  pred = clf.predict(test_X)
  print('... finished after %f secs' % (time.time() - start))
  return train_pred, pred


def nn(train_X, train_y, test_X, num_classes):
  start = time.time()
  print('\n')
  print('\n')
  print('-------------------------------------------------')
  print('Starting NN training with layers: %s ...' % ','.join(str(x) for x in FLAGS.nn_depths))
  layers = [tf.keras.layers.Dense(d, activation=tf.nn.relu) for d in
      FLAGS.nn_depths]

  model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten()] + layers + [
      tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_X, train_y, epochs=FLAGS.nn_epochs)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Train Eval ...')
  train_pred = model.predict_classes(train_X)
  print('... finished after %f secs' % (time.time() - start))
  start = time.time()
  print('Starting Eval ...')
  pred = model.predict_classes(test_X)
  print('... finished after %f secs' % (time.time() - start))
  return train_pred, pred


def compute_accuracy(pred, gt):
  return float(np.sum(np.array(pred) == np.array(gt))) / len(gt)


def main(argv):
  if FLAGS.dataset == 'toy':
    train_X, train_y, test_X, test_y, num_classes = get_toy_dataset()
  elif FLAGS.dataset == 'mnist':
    train_X, train_y, test_X, test_y, num_classes = get_mnist()

  train_pred = None

  if FLAGS.method == 'knn':
    pred = knn(train_X, train_y, test_X)
  elif FLAGS.method == 'svm':
    train_pred, pred = svm(train_X, train_y, test_X)
  elif FLAGS.method == 'tree':
    pred = tree(train_X, train_y, test_X)
  elif FLAGS.method == 'boosting':
    pred = boosting(train_X, train_y, test_X)
  elif FLAGS.method == 'nn':
    train_pred, pred = nn(train_X, train_y, test_X, num_classes)

  if train_pred is not None:
    print('Train Accuracy: %f' % compute_accuracy(train_pred, train_y))

  print('Accuracy: %f' % compute_accuracy(pred, test_y))


if __name__ == '__main__':
  app.run(main)
