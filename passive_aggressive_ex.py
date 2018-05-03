import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    dataset = SimpleDataset(is_confused=True, x=3, y=5)
    feature_vec = dataset.dataset.ix[:, "x1":"x2"]
    feature_vec["b"] = np.ones(dataset.dataset.shape[0])
    feature_vec = feature_vec.as_matrix()

    y = dataset.dataset.ix[:,"label"]
    y = y.as_matrix()

    model = PassiveAggressive()
    model_one = PassiveAggressiveOne(0.1)
    model_two = PassiveAggressiveTwo(0.1)

    plt.style.use('seaborn-colorblind')
    plt.xlim([dataset.dataset.x1.min() - 0.1, dataset.dataset.x1.max() + 0.1])
    plt.ylim([dataset.dataset.x2.min() - 0.1, dataset.dataset.x2.max() + 0.1])

    line_x = np.array(range(-10, 10, 1))
    line_y = line_x*0
    line_PA, = plt.plot(line_x, line_y, c="#2980b9", label="PA")
    line_PA_one, = plt.plot(line_x, line_y, c="#e74c3c", label="PA-1")
    line_PA_two, = plt.plot(line_x, line_y, c="#f1c40f", label="PA-2")
    
    for i in range(len(y)):
        plt.scatter(x=dataset.dataset.x1[i], y=dataset.dataset.x2[i], c=cm.cool(dataset.dataset.label[i]), alpha=0.5)
        model.fit(feature_vec[i], y[i])
        model_one.fit(feature_vec[i], y[i])
        model_two.fit(feature_vec[i], y[i])

        a, b, c = model.w
        line_y = (a*line_x + c)/(-b)
        line_PA.set_data(line_x, line_y)
        
        a, b, c = model_one.w
        line_y = (a*line_x + c)/(-b)
        line_PA_one.set_data(line_x, line_y)
        
        a, b, c = model_two.w
        line_y = (a*line_x + c)/(-b)
        line_PA_two.set_data(line_x, line_y)
        
        plt.legend(handles=[line_PA, line_PA_one, line_PA_two])
        plt.pause(0.005)


    #plt.show()


if __name__ == '__main__':
    main()
