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
    train_dataset = SimpleDataset(total_num=1000, is_confused=True,  x=3, y=5, seed=1)
    test_dataset  = SimpleDataset(total_num=100 , is_confused=False, x=3, y=5, seed=2)

    model_PA  = PassiveAggressive()
    model_one = PassiveAggressiveOne(0.1)
    model_two = PassiveAggressiveTwo(0.1)

    plt.style.use('seaborn-colorblind')
    plt.xlim([train_dataset.dataset.x1.min() - 0.1, train_dataset.dataset.x1.max() + 0.1])
    plt.ylim([train_dataset.dataset.x2.min() - 0.1, train_dataset.dataset.x2.max() + 0.1])

    line_x = np.array(range(-10, 10, 1))
    line_y = line_x*0
    line_PA,     = plt.plot(line_x, line_y, c="#2980b9", label="PA")
    line_PA_one, = plt.plot(line_x, line_y, c="#e74c3c", label="PA-1")
    line_PA_two, = plt.plot(line_x, line_y, c="#f1c40f", label="PA-2")
    
    valid_result_sample = []
    accuracies_PA = []
    
    for i in range(len(train_dataset.y)):
        plt.scatter(x=train_dataset.dataset.x1[i], y=train_dataset.dataset.x2[i], c=cm.cool(train_dataset.dataset.label[i]), alpha=0.5)
        model_PA.fit(train_dataset.feature_vec[i], train_dataset.y[i])
        model_one.fit(train_dataset.feature_vec[i], train_dataset.y[i])
        model_two.fit(train_dataset.feature_vec[i], train_dataset.y[i])
            
        # print(test_dataset.valid_training_result(model_PA))
        # valid_result_sample.append(i)
        # accuracies_PA.append(test_dataset.valid_training_result(model_PA))
        
        a, b, c = model_PA.w
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


if __name__ == '__main__':
    main()
