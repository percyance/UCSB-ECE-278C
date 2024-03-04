import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse
from tqdm import tqdm

file_path = 'F:/desktop/results/complex_wave.npy'


parser = argparse.ArgumentParser(description="choose time domain or frequency domain")
group = parser.add_mutually_exclusive_group()  # 创建互斥组
group.add_argument('--time_domain', action='store_true', help='show the magnitude')
group.add_argument('--fre_domain', action='store_true', help='show the FFT')
parser.add_argument('--stack', action='store_true', help='whether stack the output')
args = parser.parse_args()


complex_wave = np.load(file_path)
temp = np.zeros((241, 241), dtype=complex)
stack = np.zeros((241, 241), dtype=complex)
i = 0

for cw in tqdm(complex_wave):

    if args.time_domain:
        temp = cw
        if args.stack:
            stack += temp
            temp = stack
    elif args.fre_domain:
        fft = fftshift(fft2(cw))
        temp = fft
        if args.stack:
            stack += temp
            temp = stack
    
    plt.figure(figsize=(12, 5))
    plt.imshow(abs(temp), extent=(-30, 30, -30, 30))
    plt.title('')
    # plt.colorbar()
    plt.savefig(f'F:/desktop/fft_stack/complex_wave_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()
    i += 1

