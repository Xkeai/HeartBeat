import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the log
perf_log = pd.read_csv("../sessions/session_180501_1436/log_180501_1436.csv")


# Building the graph
fig, (ax1, ax2) = plt.subplots(1, 2)

# The graph for the loss
ax1.semilogy(perf_log["train_step"],
             perf_log["train_loss"],
             label="Training Loss")
ax1.semilogy(perf_log["train_step"],
             perf_log["valid_loss"],
             label="Validation Loss")
ax1.legend()
ax1.grid()
ax1.set(xlabel="Time steps", ylabel="Loss")

# The graph for the accuracy
ax2.plot(perf_log["train_step"],
         perf_log["train_accuracy"],
         label="Training Accuracy")

ax2.plot(perf_log["train_step"],
         perf_log["valid_accuracy"],
         label="Validation Accuracy")

ax2.legend()
ax2.grid()
ax2.set(xlabel="Time steps", ylabel="Accuracy")

plt.show()
