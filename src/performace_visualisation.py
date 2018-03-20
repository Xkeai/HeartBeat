from ggplot import *

import pandas as pd

perf_log = pd.read_csv("../logs/log_180316_0843.csv")
perf_log_tidy = pd.melt(perf_log,
                        id_vars=["train_step", "epoch"],
                        value_vars=["train_loss", "valid_loss"])


p = ggplot(perf_log_tidy, aes(x="train_step", y="value", colour="variable")) +\
    geom_line()

print(p)
