'''

Create a optimizer

'''

from keras import optimizers


class OptimizerCreator(object):
    def __init__(self, OPTIMIZER,  learning_rate, momentum):
        self.OPTIMIZER = OPTIMIZER
        self.learning_rate = learning_rate
        self.momentum = momentum

    def pick_opt(self):
        if self.OPTIMIZER is 'sgd':
            return self.create_sgd()

        if self.OPTIMIZER is 'adadelta':
            return self.create_adadelta(), self.learning_rate, self.momentum

        else:
            raise NameError('Given optimizer is not implemented.')

    def create_sgd(self):
        if self.learning_rate is None: self.learning_rate = 0.01
        if self.momentum is None: self.momentum = 0.9

        return optimizers.SGD(lr=self.learning_rate, momentum=self.momentum), self.learning_rate, self.momentum

    def create_adadelta(self):
        if self.learning_rate is None: self.learning_rate = 1
        self.momentum = None

        return optimizers.Adadelta(lr=self.learning_rate, rho=0.95)


