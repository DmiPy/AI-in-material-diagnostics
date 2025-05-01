from os.path import dirname, join as pjoin
import soundfile as sf
import glob, os
import pandas as pd
from scipy.signal import stft, get_window
import numpy as np
import matplotlib.pyplot as plt

# 1.Read all the files

os.chdir("Z01")

wavfiles_z01 = []

for file in glob.glob("*.wav"):
	wavfiles_z01.append(file) # getting all audiofile names from z01

os.chdir("..")

wavfiles_z05 = []

os.chdir("Z05")

for file in glob.glob("*.wav"):
	wavfiles_z05.append(file) # getting all audiofile names from z05

os.chdir("..")

print(wavfiles_z01)
data, samplerate = sf.read(f"Z01/{wavfiles_z01[0]}")
print("loaded", data.shape, "at", samplerate, "Hz")

# 2.Save the files to the dataFrame
arr1 = np.array(wavfiles_z01)
arr2 = np.array(wavfiles_z05)

wavfiles = np.concatenate((arr1,arr2))

def getDataFromFileName(array):
	specs = []
	mIDs = []
	timeFrames = []
	rIDs = []
	sIDs = []
	for element in array:
		arrOfData = element.split('_')
		specs.append(arrOfData[0])
		mIDs.append(arrOfData[4])
		timeFrames.append(arrOfData[5])
		rIDs.append(arrOfData[2])
		sIDs.append((arrOfData[8].split("."))[0])
	return (specs,mIDs,timeFrames,rIDs,sIDs)


specs,mIDs,timeFrames,rIDs,sIDs = getDataFromFileName(wavfiles)

def getSignalData(array):
	signals = []
	for fn in array:

		folder = "Z01" if "Z01" in fn else "Z05"
		filepath = os.path.join(folder, fn)

		sig, samplerate = sf.read(filepath)
		signals.append(sig)
	return (signals,samplerate)

signals , fs = getSignalData(wavfiles)

data = {
	"fn": wavfiles,				# data names(Dateinamen)
	"sig": signals,				# signal data(Signaldaten)
	"spec":specs,				# sample names
	"mID":mIDs,					# measurement ID (MessungsID)
	"time":timeFrames,			# Zeitstempel (time)
	"rID":rIDs,					# record ID (Aufnahme ID)
	"sID":sIDs,					# sensor ID
}

df = pd.DataFrame(data)

# 3. Calculate the short-term FFT for each signal:

window_type = 'hann'
nperseg = 256					# length of window (number of FFT points)
noverlap = 128					# Number of points to overlap between segments

# Compute the Short-Time Fourier Transform (STFT) of a signal
# @param signal: 1D NumPy array, the time-domain audio signal (e.g., microphone measurement)
# @param fs: Sampling frequency of the signal in Hz
# @param window_type: Type of window function to apply (e.g., 'hann', 'hamming') to reduce spectral leakage
# @param nperseg: Number of samples per segment (window length)
# @param noverlap: Number of samples each segment overlaps
# @return:
#   f - array of sample frequencies (y-axis of spectrogram),
#   t - array of segment times (x-axis of spectrogram)
#   np.abs(Zxx) - matrix of STFT magnitudes (frequency content over time)
def compute_stft(signal, fs, window_type, nperseg, noverlap):
	f, t, Zxx = stft(signal, fs=fs, window=window_type, nperseg=nperseg, noverlap=noverlap)
	return f, t, np.abs(Zxx)


stft_array = []
frequencies = []
times = []

for signal in df["sig"]:
    f, t, Zxx = compute_stft(signal, fs, window_type, nperseg, noverlap)
    stft_array.append(np.abs(Zxx))
    frequencies.append(f)
    times.append(t)

df["stft"] = stft_array
df["f"] = frequencies
df["t"] = times

# Spectogramms

# Convert amplitude (magnitude) values to decibels (dB)
# @param S: Amplitude values (usually output of STFT)
# @param ref: Reference value for 0 dB (default is 1.0)
# @param amin: Minimal value to avoid log(0), ensures numerical stability
# @return: dB-scaled values (logarithmic representation of amplitudes)
def amplitude_to_db(S, ref=1.0, amin=1e-10):
    return 20 * np.log10(np.maximum(amin, S / ref))
# Why use the logarithm (decibels)?
#     Human hearing and vibrations are perceived logarithmically, not linearly.
#     Without the logarithm, small values will be too pale and almost invisible on a graph.
#     The formula 20 * log10(amplitude) is the standard for converting amplitude to decibels (dB), similar to measuring loudness.
# amin is used to avoid the log(0) error

def plot_spectrogram(stft_matrix, filename, f, t, max_freq=20000):
    plt.figure(figsize=(10, 5))
    db_matrix = amplitude_to_db(stft_matrix)
    plt.pcolormesh(t, f, db_matrix, shading='gouraud', cmap='plasma')
    plt.title(f"Spektrogramm: {filename}")
    plt.ylabel('Frequenz [Hz]')
    plt.xlabel('Zeit [s]')
    plt.colorbar(label='Amplitude [dB]')
    plt.ylim(0, max_freq)
    plt.tight_layout()
    plt.show()


for i in range(10):
    plot_spectrogram(np.abs(df["stft"][i]), df["fn"][i], df["f"][i], df["t"][i])


# f, t, Zxx = compute_stft(signal, fs, window_type, nperseg, noverlap)
# plot_spectrogram(np.abs(Zxx), filename, f, t)

print(np.max(df["sig"][0]), np.min(df["sig"][0]))
