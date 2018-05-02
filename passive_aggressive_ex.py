import numpy as np
from simple_dataset import SimpleDataset
from passive_aggressive import PassiveAggressive

class PassiveAggressiveOne(PassiveAggressive):
    def __init__(self, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self)

    def calc_eta(self, loss, vec_x):
        l2_norm = vec_x.dot(vec_x)
        return min(self.c, loss/l2_norm)


class PassiveAggressiveTwo(PassiveAggressive):
    def __init__(self, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self)

    def calc_eta(self, loss, vec_x):
        l2_norm = vec_x.dot(vec_x)
        return loss/(l2_norm+1/(2*self.c))


def main():
    dataset = SimpleDataset(x=3, y=5)
    feature_vec = dataset.dataset.ix[:, "x1":"x2"]
    feature_vec["b"] = np.ones(dataset.dataset.shape[0])
    feature_vec = feature_vec.as_matrix()

    y = dataset.dataset.ix[:,"label"]
    y = y.as_matrix()

    model = PassiveAggressiveTwo(0.1)

    for i in range(len(y)):
        model.fit(feature_vec[i], y[i])

    print(model.w)
    dataset.show_result(model.w)


if __name__ == '__main__':
    main()
