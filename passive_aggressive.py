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
        if self.w is None:
            weight_dim = len(vec_feature)
            self.w = np.ones(weight_dim)

        self.update(vec_feature, y)


def main():
    train_dataset = SimpleDataset(total_num=1000, is_confused=True, x=3, y=5, seed=1)
    test_dataset  = SimpleDataset(total_num=300 , is_confused=False, x=3, y=5, seed=2)

    model = PassiveAggressive()

    for i in range(len(train_dataset.y)):
        model.fit(train_dataset.feature_vec[i], train_dataset.y[i])


    print(model.w)
    print(test_dataset.valid_training_result(model))
    train_dataset.show_result(model.w)


if __name__ == '__main__':
    main()
