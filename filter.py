import sys
import numpy as np
from tqdm import tqdm
from scipy import signal, interpolate
from scipy.io import wavfile
import matplotlib.pyplot as plt

NEW_RATE = 3000

testStart = 0
testEnd = 60 * NEW_RATE # Analyze first 60 seconds

def changeRate(sig, oldRate, newRate):
    duration = sig.shape[0] / oldRate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * newRate / oldRate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    sig = interpolator(time_new).T

    return sig

def normalize(v):
    norm = max(v)
    if norm == 0:
       return v
    return v / norm

rateTestSig, testSig = wavfile.read(sys.argv[1]) # testSig / longer audio file
rateRefSig, refSig = wavfile.read(sys.argv[2]) # refSig to be found in the other file

testSig = changeRate(testSig, rateTestSig, NEW_RATE)[:, 1]
refSig = changeRate(refSig, rateRefSig, NEW_RATE)[:, 1]

testSig = normalize(testSig)
refSig = normalize(refSig)

crossCorrelation = np.correlate(testSig[testStart:testEnd], refSig)

m = max(crossCorrelation)
print([i / NEW_RATE for i, j in enumerate(crossCorrelation) if j == m])

ticks = [x for x in range(len(testSig)) if (x / NEW_RATE) % 15 == 0]
tickLabels = [x / NEW_RATE for x in ticks]

plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.5)
plt.title("Cross-Correlation")
plt.xticks(ticks, tickLabels)
plt.plot(crossCorrelation)

plt.subplot(3,1,2)
plt.title("Test Signal")
plt.plot(testSig[:testEnd])

plt.subplot(3,1,3)
plt.title("Reference Signal")
plt.plot(refSig)
plt.show()
