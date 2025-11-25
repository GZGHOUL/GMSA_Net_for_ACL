import numpy as np
import pandas as pd
import os
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

data_path = 'Dataset/Gait/gait_train_renum_merged_A.csv'
df = pd.read_csv(data_path)

person = np.array(eval(df['features'].iloc[1]))
person = person.T
person = person[5,:]
person = person - np.mean(person)
person_fft = fft(person)


plt.plot(person_fft)
plt.show()

