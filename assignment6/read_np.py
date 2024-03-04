import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse
from tqdm import tqdm

file_path = 'F:/desktop/results/complex_wave.npy'

complex_wave = np.load(file_path)
temp = np.zeros((40, 800), dtype=complex)
stack = np.zeros((40, 800), dtype=complex)
i = 0

for cw in tqdm(complex_wave):
    
    temp += cw
    plt.figure(figsize=(12, 5))
    plt.imshow(abs(temp), extent=(-2.13,2.13,0 ,0.213))
    plt.title('')
    # plt.colorbar()
    plt.savefig(f'F:\\desktop\\results\\{i}.png', dpi=300, bbox_inches='tight')
    plt.close()
    i += 1


