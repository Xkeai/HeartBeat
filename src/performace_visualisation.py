import matplotlib.pyplot as plt
import pandas as pd

# Importing the log
perf_log = pd.read_csv("../logs/log_180419_1358.csv")


# Building the graph
fig, ax = plt.subplots()

ax.plot(perf_log["train_step"],
        perf_log["train_loss"],
        label="Training Loss")

ax.plot(perf_log["train_step"],
        perf_log["valid_loss"],
        label="Validation Loss")

plt.legend()
ax.grid()
ax.set(xlabel="Time steps", ylabel="Loss")

plt.show()
