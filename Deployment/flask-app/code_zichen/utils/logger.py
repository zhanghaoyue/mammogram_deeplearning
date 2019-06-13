import json
import logging

logging.basicConfig(level=logging.INFO, format='')


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        # len(self.entries) + 1 records the epoch number
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        # print logging info with jason format
        return json.dumps(self.entries, sort_keys=True, indent=4)
