'''

TOOLS FOR CLASSIFICATION USING LINEAR MODELS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import itertools
import functools
import numpy as np

# %% PREPROCESSING

class Polynomial_Features(object):
    '''
    Polynomial feature transformer

    Attributes
    ----------
    degree: int

    '''

    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, samples):
        '''
        Returns transformed features with augmented ones

        :param samples: raw samples

        :return: augmented matrix of samples

        '''

        if samples.ndim == 1:
            samples = samples[:, None]

        features = [np.ones(len(samples))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(samples.T, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


class Label_Transformer(object):
    '''
    Label encoder decoder

    Attributes
    ----------
    number of classes n_classes : int

    '''

    def __init__(self, n_classes:int=None):
        self.n_classes = n_classes

    @property
    def n_classes(self):
        return self.__n_classes

    @n_classes.setter
    def n_classes(self, K):
        self.__n_classes = K
        self.__encoder = None if K is None else np.eye(K)

    @property
    def encoder(self):
        return self.__encoder

    def encode(self, class_indices:np.ndarray):
        '''
        One-hot encoding

        :param class_indices: class index

        :return: One-hot encoding

        '''

        if self.n_classes is None:
            self.n_classes = np.max(class_indices) + 1

        return self.encoder[class_indices]

    def decode(self, onehot:np.ndarray):
        '''
        Decoding from one-hot

        :param onehot: one-hot vector

        :return: class index
        
        '''

        return np.argmax(onehot, axis=1)

# %% CLASSIFIERS

class Classifier():
    pass

class Least_Squares_Classifier(Classifier):
    '''
    Linear Least-Squares Classifier

    Attributes
    ----------
    weights: ndarray (d+1)

    '''

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, samples, labels):
        '''
        Train linear least-squares classifier

        :param samples: training samples
        :param labels: training labels

        :setter: learnt weights

        '''

        if labels.ndim == 1:
            labels = Label_Transformer().encode(labels)

        self.weights = np.linalg.pinv(samples) @ labels

    def predict(self, samples):
        '''
        Predict labels using linear least-squares classifier

        :param samples: test samples

        :returns: predicted labels

        '''

        return np.argmax(samples @ self.weights, axis=-1)

    def accuracy(self, samples, labels):
        '''
        Returns the confusion matrix with testing accuracies

        :param samples: training samples
        :param labels: true labels

        :return: confusion matrix

        '''

        num_samples, _ = samples.shape
        num_classes = len(np.unique(labels))

        predictions = self.predict(samples)

        confusion_mtx = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_mtx[i,j] = sum((labels==i) & (predictions==j))

        return num_classes*confusion_mtx/num_samples

class Logistic_Regression(Classifier):
    '''
    Logistic Regression

    Attributes
    ----------

    weights: ndarray (d+1)

    '''

    def __init__(self, weights=None):
        self.weights = weights

    @staticmethod
    def _activation(a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, samples, labels, max_iter=100, lr=0.01):
        '''
        Trains multiclass logistic regression

        :param samples: training samples
        :param labels: training labels
        :optional max_iter: maximum iterations of gradient descent
        :optional lr: learning rate of gradient descent

        :setter: weights

        '''

        if labels.ndim == 1:
            labels = Label_Transformer().encode(labels)
        self.n_classes = labels.shape[1]
        
        weights = np.zeros((np.size(samples, 1), self.n_classes))
        for _ in range(max_iter):
            weights_prev = np.copy(weights)
            
            prediction = self._activation(samples @ weights)
            grad = samples.T @ (prediction - labels)
            weights -= lr * grad
            
            if np.allclose(weights, weights_prev):
                break

        self.weights = weights

    def probs(self, samples):
        '''
        Computes the probabilities of samples
        belonging to each class

        :param samples: test samples

        :return: probability vector

        '''

        return self._activation(samples @ self.weights)

    def predict(self, samples):
        '''
        Predict labels for test samples

        :param samples: test samples

        :return: class labels for samples

        '''

        return np.argmax(self.probs(samples), axis=-1)

    def accuracy(self, samples, labels):
        '''
        Returns the confusion matrix with testing accuracies

        :param samples: training samples
        :param labels: true labels

        :return: confusion matrix

        '''

        num_samples, _ = samples.shape
        num_classes = len(np.unique(labels))

        predictions = self.predict(samples)

        confusion_mtx = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_mtx[i,j] = sum((labels==i) & (predictions==j))

        return num_classes*confusion_mtx/num_samples