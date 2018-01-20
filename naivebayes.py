from classificationalgo import *
import numpy as np
import scipy
from scipy import stats

"""
Klasa implementujaca naiwny klasyfikator bayesa.
"""
class NaiveBayes(ClassificationAlgorithm):

    """
    Konstruktor
    n_labels - liczba cyfr reprezentowanych przez rysunki w zbiorze
    n_features - liczba atrybutow opisujacych kazda z probek - rysunek
    """
    def __init__(self, n_labels=10, n_features=81):
        self.n_labels = np.array(n_labels)
        self.n_features = np.array(n_features)
        self.mean = np.zeros((n_labels, n_features), dtype=np.float)
        self.var = np.zeros((n_labels, n_features), dtype=np.float)
        self.pi = np.zeros(n_labels, dtype=np.float)

    """
    Metoda, ktorej wywolanie uruchamia algorytm predykcji
    data - dane, dla ktorych ma zostac przeprowadzona predykcja
    """
    def predict(self, data):
        return np.array([self.__classify(x) for x in data])

    """
    Metoda sluzaca do nauki na podstawie danych treningowych
    data - probki/rysunki do nauki
    labels - wyniki(cyfry) dla kolejnych elementow ze zbioru data
             w pierwszym argumencie
    """
    def train(self, data, labels):
        N = data.shape[0]  # licznosc zbioru treningowego
        N_l = np.array([(labels == y).sum() for y in range(self.n_labels)], dtype=np.float)

        # aktualizacja sredniej
        for y in range(self.n_labels):
            sum = np.sum(data[n] if labels[n] == y else 0.0 for n in range(N))
            self.mean[y] = sum / N_l[y]

        # aktualizacja gausowskiej wariancji
        for y in range(self.n_labels):
            sum = np.sum((data[n] - self.mean[y]) ** 2 if labels[n] == y else 0.0 for n in range(N))
            self.var[y] = sum / N_l[y]

        # aktualizacja prawd. apriori
        self.pi = N_l / N

    """
    Klasyfikacja pojedynczej probki
    x - probka do predykcji
    """
    def __classify(self, x):
        results = [self.__negative_log_likelihood(x, y) for y in range(self.n_labels)]
        return np.argmin(results)

    """
    Metoda wyliczajaca prawdopodobienstwa apriori oraz aposteriori dla
    podanej probki
    x - probka do predykcji
    y - probka do porownania
    """
    def __negative_log_likelihood(self, x, y):
        log_prior_y = -np.log(self.pi[y])
        log_posterior_x_given_y = -np.sum(
            [self.__log_gaussian(x[d], self.mean[y][d], self.var[y][d]) for d in range(self.n_features)]
        )
        return log_prior_y + log_posterior_x_given_y

    """
    Metoda wyliczajaca wartosci funkcji gestosci prawdopodobienstwa
    rozkladu lognormalnego dla podanej probki i odpowiedniej sredniej
    oraz wariancji
    x - probka
    mean - srednia
    var - wariancja
    """
    def __log_gaussian(self, x, mean, var):
        epsiron = 1.0e-5
        if var < epsiron:
            return 0.0
        # funkcja gestosci prawdopodobienstwa rozkladu lognormalnego
        return scipy.stats.norm(mean, var).logpdf(x)


"""
Classifier NaiveBayes trained in 1.4239020347595215 sec
Evaluating model ...
Accuracy: 85.62 %
confusion matrix:
[[ 933    0   32    0    0    3    0    0   12    0]
 [   0 1056   45    0    2    0   13   18    1    0]
 [   0    0 1028    0    0    0    0    3    0    1]
 [   3    0  372  519    0   30    0   12   66    8]
 [   1    1   51    0  855    0   14   22    9   29]
 [   5    0   25    4    0  816    1    2   37    2]
 [  26    1   37    0    4    9  849    0   32    0]
 [   0    1   87    0    0    0    0  917   13   10]
 [  15    2   83    0    3    4    6   11  838   12]
 [  12    2   32    0    0    3    1  174   34  751]]
Classifier NaiveBayes estimated in 5931.262767791748 sec
"""
