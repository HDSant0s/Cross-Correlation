import sys
import numpy as np
from tqdm import tqdm
from scipy import signal, interpolate
from scipy.io import wavfile
import matplotlib.pyplot as plt

NEW_RATE = 3000

TEST_START = 0
TEST_END = 60 * NEW_RATE # Analyze first 60 seconds

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

rateTestSig, testSig = wavfile.read(sys.argv[1]) # Test signal (longer)
rateRefSig, refSig = wavfile.read(sys.argv[2]) # Reference signal to be found in the other file

# Downsample the signals for quicker analysis
testSig = changeRate(testSig, rateTestSig, NEW_RATE)[:, 1]
refSig = changeRate(refSig, rateRefSig, NEW_RATE)[:, 1]

# Normalize the signals
testSig = normalize(testSig)
refSig = normalize(refSig)

# Calculate Cross-Correlation
crossCorrelation = np.correlate(testSig[TEST_START:TEST_END], refSig)

# Get maximum value and location
m = max(crossCorrelation)
print([i / NEW_RATE for i, j in enumerate(crossCorrelation) if j == m])

# Set ticks for x-Axis
ticks = [x for x in range(len(testSig)) if (x / NEW_RATE) % 15 == 0]
tickLabels = [x / NEW_RATE for x in ticks]

# Create plots
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
