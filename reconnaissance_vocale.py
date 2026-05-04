import torch 
import numpy as np
import matplotlib 

def passe_bande(signal):
    """coupe les frequences pour ne garder que le spectre de la voix 20-2000 Hz"""
    tf_signal = np.fft.fft(signal)
    tf_signal[2000:]=0 
    tf_signal[:20]=0
    return np.fft.ifft(tf_signal)
