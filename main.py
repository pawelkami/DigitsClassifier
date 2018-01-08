import struct
import os.path
import numpy as np
import cv2
import pickle
from naivebayes import *
from decisiontree import *
import time

PICKLE_TRAINING_SET = "training_set.pickle"
PICKLE_TESTING_SET = "testing_set.pickle"
IMAGE_HEIGHT = IMAGE_WEIGHT = 28


def read_MNIST(howMany, train=True):
    """
        Method to read the .idxY-ubyte files downloaded form http://yann.lecun.com/exdb/mnist/
        howMany ->  used to determine how many labels/images have to be loaded. This method is
                    setup so that it will always load from the first image to the howManyth.
        train ->    used to determine if the training or the testing data has to be loaded.
    """

    if train:
        if os.path.isfile(PICKLE_TRAINING_SET):
            with open(PICKLE_TRAINING_SET, 'rb') as file:
                image_list, lable_list = pickle.load(file)
                return image_list, lable_list
        images = open('train-images.idx3-ubyte', 'rb')
        labels = open('train-labels.idx1-ubyte', 'rb')
        print('Reading Training Data')
    else:
        if os.path.isfile(PICKLE_TESTING_SET):
            with open(PICKLE_TESTING_SET, 'rb') as file:
                image_list, lable_list = pickle.load(file)
                return image_list, lable_list
        images = open('t10k-images.idx3-ubyte', 'rb')
        labels = open('t10k-labels.idx1-ubyte', 'rb')
        print('Reading Test Data')

    # ----------------------------- Loading the Labels ------------------------------------
    # Reading the magic number and the number of items in the file
    print('\nReading labels:')
    magicNumber = labels.read(4)
    print('Magic number: ', struct.unpack('>I', magicNumber)[0])
    numberOfItems = labels.read(4)
    print('Number of Items in MNIST File: ', struct.unpack('>I', numberOfItems)[0])
    if howMany > struct.unpack('>I', numberOfItems)[0]:
        howMany = struct.unpack('>I', numberOfItems)[0]
    howMany = howMany
    print('Number of Files to read: ', howMany)

    # reading the labels, depending on howMany should be read. (Every byte is a label)
    lable_list = []
    byte = labels.read(1)
    while len(byte) > 0 and len(lable_list) < howMany:
        lable_list.append(struct.unpack('>B', byte)[0])
        byte = labels.read(1)
    labels.close()
    i = 10
    if howMany < 10:
        i = howMany / 2
    print('First ' + str(i) + ' labels: ', lable_list[:i])
    lable_list = lable_list

    # ----------------------------- Loading the Images ------------------------------------
    # reading the magic number, number of items, number of rows and columns
    print('\nReading Images:')
    magicNumber = images.read(4)
    print('Magic number: ', struct.unpack('>I', magicNumber)[0])
    numberOfItems = images.read(4)
    print('Number of Items in MNIST File: ', struct.unpack('>I', numberOfItems)[0])
    numOfRows = images.read(4)
    print('Number of rows: ', struct.unpack('>I', numOfRows)[0])
    numOfCols = images.read(4)
    print('Number of columns: ', struct.unpack('>I', numOfCols)[0])
    selfnumOfCols = struct.unpack('>I', numOfCols)[0]
    selfnumOfRows = struct.unpack('>I', numOfRows)[0]

    print('')

    if howMany > 10000:
        blub = int(howMany / 25)
    else:
        blub = int(howMany / 10)

    # reading the images, depending on howMany. (Every Byte is a pixel)
    image_list = []
    for i in range(howMany):
        if i > 0 and i % blub == 0:
            print('Loaded ' + str(i / float(howMany) * 100) + '% of the images')
        image = []
        for j in range(struct.unpack('>I', numOfRows)[0] * struct.unpack('>I', numOfCols)[0]):
            x = struct.unpack('>B', images.read(1))[0]
            image.append(x)
        image_list.append(image)
    images.close()

    if train:
        with open(PICKLE_TRAINING_SET, 'wb') as file:
            pickle.dump((image_list, lable_list), file)
    else:
        with open(PICKLE_TESTING_SET, 'wb') as file:
            pickle.dump((image_list, lable_list), file)

    return image_list, lable_list


def grayscale_pixelarray_to_np(training_image):
    pixels = np.array(training_image, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    return pixels


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * IMAGE_HEIGHT * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (IMAGE_HEIGHT, IMAGE_HEIGHT), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def hu_moments(img):
    moments = cv2.HuMoments(cv2.moments(img)).flatten()
    for i in range(len(moments)):
        moments[i] = np.float32(moments[i])
    # print(moments)
    return moments


def get_hog():
    winSize = (28, 28)
    blockSize = (14, 14)
    blockStride = (7, 7)
    cellSize = (14, 14)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
        # return mosaic(75, vis)


if __name__ == '__main__':
    training = True
    print('Reading MNIST Training set')
    training_set, training_labels = read_MNIST(60000, training)
    training_set_np = list(map(grayscale_pixelarray_to_np, training_set))
    training_labels_np = np.array(training_labels, dtype='int')
    training_labels_np = np.squeeze(training_labels)

    hog = get_hog()

    print('Calculating HoG descriptors')
    hog_descriptors = []
    for img in training_set_np:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Reading MNIST Testing set')
    testing_set, testing_labels = read_MNIST(10000, not training)
    testing_set_np = list(map(grayscale_pixelarray_to_np, testing_set))
    testing_labels_np = np.array(testing_labels, dtype='int')
    testing_labels_np = np.squeeze(testing_labels_np)

    hog_descriptors_test = []
    for img in testing_set_np:
        hog_descriptors_test.append(hog.compute(img))
    hog_descriptors_test = np.squeeze(hog_descriptors_test)

    algorithms = [SVM,
                  NaiveBayes,
                  # DecisionTree
                 ]  # odkomentować po zaimplementowaniu

    for a in algorithms:
        model = a()
        start = time.time()
        model.train(hog_descriptors, training_labels_np)
        end = time.time()
        print("Classifier {} trained in {} sec".format(model.__class__.__name__, end - start))

        print('Evaluating model ... ')
        start = time.time()
        evaluate_model(model, testing_set_np, hog_descriptors_test, testing_labels_np)
        end = time.time()
        print("Classifier {} estimated in {} sec".format(model.__class__.__name__, end - start))

