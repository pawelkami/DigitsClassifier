from abc import abstractmethod, ABC


class ClassificationAlgorithm(ABC):
    @abstractmethod
    def train(self, samples, responses):
        pass

    @abstractmethod
    def predict(self, samples):
        pass

