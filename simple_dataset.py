import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleDataset:
    def __init__(self, total_num=1000, is_confused=False, confuse_bin=30, x=5, y=3, seed=1):
        np.random.seed(seed)
        feature = np.random.randn(total_num, 2)
        self.dataset = pd.DataFrame(feature, columns=["x1", "x2"])
        self.dataset["label"] = self.dataset.apply(lambda row : 1 if (x*row.x1 + y*row.x2 - 1)>0 else -1, axis=1)

        if is_confused:
            def make_data_confused(data):
                label = -1*data.label if (data.name % confuse_bin) == 0 else data.label
                return label

            self.dataset["label"] = self.dataset.apply(make_data_confused, axis=1)


    def show_dataset(self):
        plt.style.use('ggplot')
        plt.scatter(x=self.dataset.x1, y=self.dataset.x2, c=self.dataset.label, alpha=0.5)

        plt.xlim([self.dataset.x1.min() - 0.1, self.dataset.x1.max() + 0.1])
        plt.ylim([self.dataset.x2.min() - 0.1, self.dataset.x2.max() + 0.1])

        plt.show()


    def show_result(self, vec_weight):
        plt.style.use('ggplot')
        plt.scatter(x=self.dataset.x1, y=self.dataset.x2, c=self.dataset.label, alpha=0.5)

        a, b, c = vec_weight
        x = np.array(range(-10, 10, 1))
        y = (a*x + c)/(-b)
        plt.plot(x, y, alpha = 0.3)

        plt.xlim([self.dataset.x1.min() - 0.1, self.dataset.x1.max() + 0.1])
        plt.ylim([self.dataset.x2.min() - 0.1, self.dataset.x2.max() + 0.1])

        plt.show()
