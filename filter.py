import sys, os
import numpy as np
from tqdm import tqdm
from scipy import signal, interpolate
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Tick interval for the plots
INTERVAL = 30
RATE = 0

# Change the sample rates of the audio files
def changeRate(sig, oldRate, newRate):
    duration = sig.shape[0] / oldRate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * newRate / oldRate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    sig = interpolator(time_new).T

    return sig

# Normalize the signals for clearer spike
def normalize(v):
    norm = max(v)
    if norm == 0:
       return v
    return v / norm

# Reads mp3 and returns the signal (samples) and sample rate
def readMp3(filename):
    audio = AudioSegment.from_mp3(filename)
    audio.export("temp.wav", format='wav')

    signal, rate = wavfile.read("temp.wav")
    os.remove("temp.wav")

    return signal, rate

# Args: 1. Test signal file, 2. Reference signal
if len(sys.argv) < 3:
    print("Syntax: testsignal referencesignal")
    sys.exit()

# Load test signal (longer)
if sys.argv[1].endswith(".wav"):
    rateTestSig, testSig = wavfile.read(sys.argv[1])
elif sys.argv[1].endswith(".mp3"):
    rateTestSig, testSig = readMp3(sys.argv[1])
else: print("File format not supported!")

# Load reference signal (known sound, shorter)
if sys.argv[2].endswith(".wav"):
    rateRefSig, refSig = wavfile.read(sys.argv[2])
elif sys.argv[2].endswith(".mp3"):
    rateRefSig, refSig = readMp3(sys.argv[2])
else: print("File format not supported!")

print("Files loaded successfully!")

# Match sample rates
if rateRefSig > rateTestSig:
    refSig = changeRate(refSig, rateRefSig, rateTestSig)
elif rateTestSig > rateRefSig:
    testSig = changeRate(testSig, rateTestSig, rateRefSig)

RATE = rateTestSig

# Convert to single channel
testSig = testSig[:, 0]
refSig = refSig[:, 0]

# Normalize the signals
testSig = normalize(testSig)
refSig = normalize(refSig)

print(len(testSig))

# Calculate Cross-Correlation
crossCorrelation =  normalize(signal.correlate(testSig, refSig, mode="valid")) # np.correlate(testSig, refSig)

print(len(crossCorrelation))

# Get peaks
peaks, properties = signal.find_peaks(crossCorrelation, distance=len(refSig), prominence=0.5)

# Set ticks for x-Axis
ticks = []
tickLabels = []
for x in range(len(testSig)):
    if ((x / RATE) % INTERVAL == 0):
        ticks.append(x)
        tickLabels.append(x / RATE)

# Create plots
plt.subplot(3,1,1)
plt.subplots_adjust(hspace=0.6)
plt.title("Cross-Correlation")
plt.xticks(ticks, tickLabels)
plt.plot(crossCorrelation)

plt.subplot(3,1,2)
plt.title("Test Signal")
plt.xticks(ticks, tickLabels)
plt.plot(testSig)
for x in peaks:
    xEnd = x+len(refSig)
    print("Start: " + str(x / RATE) + " (" + str(x) + " samples)")
    print("End: " + str(xEnd / RATE) + " (" + str(xEnd) + " samples)")
    plt.axvline(x=x, linestyle="--", color="r")
    plt.axvline(x=xEnd, linestyle="--", color="r")

plt.subplot(3,1,3)
plt.title("Reference Signal")
plt.xticks(ticks, tickLabels)
plt.plot(refSig)
plt.show()
