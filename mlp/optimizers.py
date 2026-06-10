# SGD e opcionais

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params:
            params[key] -= self.learning_rate * grads[key]


class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def update(self, params, grads):

        for key in params:

            if key not in self.velocity:
                self.velocity[key] = 0

            self.velocity[key] = (
                self.beta * self.velocity[key]
                - self.learning_rate * grads[key]
            )

            params[key] += self.velocity[key]