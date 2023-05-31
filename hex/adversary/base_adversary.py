from abc import abstractmethod

from hex.qmodels.q_model import QModel


class BaseAdversary(object):

    def __init__(self):
        pass

    @abstractmethod
    def init(self, q_learner):
        pass

    @abstractmethod
    def update(self, q_learner, epoch, showPlot=False, random_start=False):
        pass

    @abstractmethod
    def get_action(self, state, q_learner):
        pass