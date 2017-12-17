from time import gmtime, strftime
import csv


def getLogName():
    return "log_" + strftime("%y%m%d_%H%M", gmtime()) + ".csv"


class LogWriter:

    def __init__(self, fname, fieldnames):
        self.fname = fname
        self.fieldnames = fieldnames
        with open(self.fname, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def addEntry(self, entry):
        with open(self.fname, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(entry)

    def addEntries(self, entries):
        with open(self.fname, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(entries)
