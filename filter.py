import sys
import numpy as np
from tqdm import tqdm
from scipy import signal, interpolate
from scipy.io import wavfile
import matplotlib.pyplot as plt

NEW_RATE = 3000

def changeRate(song, oldRate, newRate):
    duration = song.shape[0] / oldRate

    time_old  = np.linspace(0, duration, song.shape[0])
    time_new  = np.linspace(0, duration, int(song.shape[0] * newRate / oldRate))

    interpolator = interpolate.interp1d(time_old, song.T)
    song = interpolator(time_new).T

    return song

def normalize(v):
    norm = max(v)
    if norm == 0:
       return v
    return v / norm

rateSong, song = wavfile.read(sys.argv[1]) # Song / longer audio file
rateElement, element = wavfile.read(sys.argv[2]) # Element to be found in the other file

song = changeRate(song, rateSong, NEW_RATE)[:, 1]
element = changeRate(element, rateElement, NEW_RATE)[:, 1]

song = normalize(song)
element = normalize(element)

testStart = 0
testEnd = 60 * NEW_RATE # Analyze first 60 seconds

crossCorrelation = np.correlate(song[testStart:testEnd], element)

# for sample, value in enumerate(crossCorrelation):
#     crossCorrelation[sample] = [sample / NEW_RATE, value]

m = max(crossCorrelation)
print([i / NEW_RATE for i, j in enumerate(crossCorrelation) if j == m])

ticks = [x for x in range(len(song)) if (x / NEW_RATE) % 15 == 0]
tickLabels = [x / NEW_RATE for x in ticks]

plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.5)
plt.title("Cross-Correlation")
plt.xticks(ticks, tickLabels)
plt.plot(crossCorrelation)

plt.subplot(3,1,2)
plt.title("Test Signal")
plt.plot(song[:testEnd])

plt.subplot(3,1,3)
plt.title("Reference Signal")
plt.plot(element)
plt.show()
