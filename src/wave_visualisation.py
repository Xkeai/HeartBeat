import numpy as np
import pandas as pd
from ggplot import *


from timingDataset import timingDataset


dataset = timingDataset("../data/set_a_timing.csv")

data_diff_0, label_diff_0 = dataset.next_batch_train()
dataset.reset()
data_diff_1, label_diff_1 = dataset.next_batch_train(diff_n=1)


print(data_diff_0.shape)
print(data_diff_1.shape)


df_0 = pd.DataFrame(data_diff_0[:, :, 0].transpose(), columns=["wave"])
df_0["index"] = df_0.index
df_1 = pd.DataFrame(data_diff_1[:, :, 0].transpose(), columns=["wave"])
df_1["index"] = df_1.index


p0 = ggplot(df_0, aes(x="index", y="wave")) +\
    geom_line()

p1 = ggplot(df_1, aes(x="index", y="wave")) +\
    geom_line()


print(p0)
print(p1)
