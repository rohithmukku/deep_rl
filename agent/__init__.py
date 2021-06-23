from abc import abstractmethod

class Agent(object):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self):
        pass
