import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if needed

import matplotlib.pyplot as plt
from collections import deque
import threading
import time

class RealTimeGraph:
    def __init__(self, max_len=200):
        self.values = deque([0], maxlen=max_len)
        self.timestamps = deque([0], maxlen=max_len)
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_plot)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def update(self, label):
        with self.lock:
            t = time.time() - self.start_time
            self.timestamps.append(t)

            if label == "REAL":
                self.values.append(1)
            elif label == "FAKE":
                self.values.append(-1)
            else:
                self.values.append(0)

    def _update_plot(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

        while self.running:
            with self.lock:
                ax.clear()
                if len(self.timestamps) > 1:
                    ax.plot(self.timestamps, self.values, color='blue', linewidth=2)
                    ax.fill_between(self.timestamps, self.values, 0,
                                    where=[v > 0 for v in self.values],
                                    interpolate=True, color='green', alpha=0.4, label='REAL')
                    ax.fill_between(self.timestamps, self.values, 0,
                                    where=[v < 0 for v in self.values],
                                    interpolate=True, color='red', alpha=0.4, label='FAKE')
                ax.set_ylim(-1.5, 1.5)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Detection")
                ax.set_title("Real vs Fake Waveform Over Time")
                ax.grid(True)
                ax.legend(loc="upper right")
            plt.pause(0.1)

        plt.ioff()
        plt.close()

        