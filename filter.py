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

# crossCorrelation = []
# for i in tqdm(range(testStart, testEnd - len(element) + 1)): # Sliding correlation
#     testSamples = song[i:len(element) + i]
#
#     res = sum(np.multiply(testSamples, element))
#     crossCorrelation.append(res)

m = max(crossCorrelation)
print([i / NEW_RATE for i, j in enumerate(crossCorrelation) if j == m])
#
# crossCorrelation = [float(i)/m for i in crossCorrelation if i >= 0]

plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.5)
plt.title("Cross-Correlation")
plt.plot(crossCorrelation)

plt.subplot(3,1,2)
plt.title("Test Signal")
plt.plot(song[:testEnd])

plt.subplot(3,1,3)
plt.title("Reference Signal")
plt.plot(element)
plt.show()
