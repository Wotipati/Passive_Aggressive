import numpy as np
from simple_dataset import SimpleDataset

class PassiveAggressive:
    def __init__(self):
        self.t = 0
        self.w = None

    def L_hinge(self, vec_x, y):
        return max([0, 1-y*self.w.dot(vec_x)])

    def calc_eta(self, loss, vec_x):
        l2_norm = vec_x.dot(vec_x)
        return loss/l2_norm

    def update(self, vec_x, y):
        loss = self.L_hinge(vec_x, y)
        eta = self.calc_eta(loss, vec_x)
        self.w += eta*y*vec_x
        self.t += 1

    def fit(self, vec_feature, y):
        weight_dim = len(vec_feature)
        if self.w is None:
            self.w = np.ones(weight_dim)

        self.update(vec_feature, y)


def main():
    dataset = SimpleDataset()
    feature_vec = dataset.dataset.ix[:, "x1":"x2"]
    feature_vec["b"] = np.ones(dataset.dataset.shape[0])
    feature_vec = feature_vec.as_matrix()

    y = dataset.dataset.ix[:,"label"]
    y = y.as_matrix()

    model = PassiveAggressive()

    for i in range(len(y)):
        model.fit(feature_vec[i], y[i])

    print(model.w)
    dataset.show_result(model.w)


if __name__ == '__main__':
    main()
