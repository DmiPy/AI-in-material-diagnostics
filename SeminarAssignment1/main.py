from os.path import dirname, join as pjoin
import soundfile as sf
import glob, os
import scipy.io
import numpy
import sys

os.chdir("Z01")

wavfiles_z01 = []

for file in glob.glob("*.wav"):
	wavfiles_z01.append(file)

os.chdir("..")

wavfiles_z05 = []

os.chdir("Z05")

for file in glob.glob("*.wav"):
	wavfiles_z05.append(file)

os.chdir("..")


data, samplerate = sf.read(f"Z01/{wavfiles_z01[0]}")
print("loaded", data.shape, "at", samplerate, "Hz")