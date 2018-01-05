import struct
import os.path
import numpy as np
import cv2
import pickle

PICKLE_TRAINING_SET = "training_set.pickle"
IMAGE_HEIGHT = IMAGE_WEIGHT = 28


def read_MNIST(howMany, train=True):
    """
        Method to read the .idxY-ubyte files downloaded form http://yann.lecun.com/exdb/mnist/
        howMany ->  used to determine how many labels/images have to be loaded. This method is
                    setup so that it will always load from the first image to the howManyth.
        train ->    used to determine if the training or the testing data has to be loaded.
    """
    if os.path.isfile(PICKLE_TRAINING_SET):
        with open(PICKLE_TRAINING_SET, 'rb') as file:
            image_list, lable_list = pickle.load(file)
            return image_list, lable_list

    if train:
        images = open('train-images.idx3-ubyte', 'rb')
        labels = open('train-labels.idx1-ubyte', 'rb')
        print('Reading Training Data')
    else:
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

    with open(PICKLE_TRAINING_SET, 'wb') as file:
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


if __name__ == '__main__':
    training = True
    print('Reading MNIST')
    # first read the training data
    training_set, training_labels = read_MNIST(60000, training)

    training_set_np = list(map(grayscale_pixelarray_to_np, training_set))

    hog = get_hog()

    print('Calculating HoG descriptors')
    hog_descriptors = []
    for img in training_set_np:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
