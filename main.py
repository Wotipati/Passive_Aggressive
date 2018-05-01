import numpy as np
from simple_dataset import SimpleDataset
from passive_aggressive import PassiveAggressive

dataset = SimpleDataset()
feature_vec = dataset.dataset.ix[:, "x":"y"]
feature_vec["b"] = np.ones(dataset.dataset.shape[0])
feature_vec = feature_vec.as_matrix()

y = dataset.dataset.ix[:,"label"]
y = y.as_matrix()

model = PassiveAggressive()

for i in range(len(y)):
    model.fit(feature_vec[i], y[i])

print(model.w)
dataset.show_result(model.w)
