# %matplotlib qt
import matplotlib.pyplot as plt

def show_data_hist(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()
