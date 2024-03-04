import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse
from tqdm import tqdm
from scipy.io import loadmat

def generate_receivers_locations(y_offset, span):
    delta_x = 0.0213/4
    num_points = int(span / delta_x)
    x_positions = np.linspace(-span/2, span/2, num_points)
    scatter_locations = [(x, y_offset) for x in x_positions]
    # print(scatter_locations)
    return scatter_locations
    
def generate_source_region(X_length, Y_length):
    delta_x = 0.0213/4
    num_points_x = int(X_length / delta_x)
    num_points_y = int(Y_length / delta_x)
    x_positions = np.linspace(-X_length/2, X_length/2, num_points_x)
    y_positions = np.linspace(0, Y_length, num_points_y)
    source_locations = [(x, y) for y in y_positions for x in x_positions]
    # print(source_locations)
    return source_locations

def calculate():

    X_length = 4.26
    Y_length = 0.213
    y_offset = 0
    span = 4.26
    mat_file_path = './gpr_data/gpr_data.mat'
    data = loadmat(mat_file_path)
    F = data['F']
    f = data['f']
    da = data['da']

    source_locations = generate_source_region(X_length, Y_length)

    num_points_x = int(X_length / (0.0213/4))
    num_points_y = int(Y_length / (0.0213/4))
    complex_wave = []
    # G = np.zeros((128,200),dtype=complex)
    if args.part_B:
        for receive in tqdm(range(200)):
            h_whole = np.zeros((num_points_y, num_points_x), dtype=complex)
            for idx, (x_s, y_s) in enumerate(source_locations):
                y_idx = idx // num_points_x
                x_idx = idx % num_points_x
                h = 0
                for n in range(128):
                    lambda_n = ((3e8/np.sqrt(6)) / f[0,n])
                    r_g = np.sqrt((da[0, receive] - 0)**2 + (0 - 0)**2)
                    x = F[n, receive].conjugate() 
                    r_s = np.sqrt((x_s - (da[0, receive] - 2.13))**2 + (y_s - 0)**2)
                    h_n = x * np.exp(-1j * 2 * np.pi * (2 * r_s) / lambda_n)
                    h += h_n
                h_whole[y_idx, x_idx] = h
            complex_wave.append(h_whole)
        return complex_wave
    

    if args.part_B:
        for n in tqdm(range(128)):
            lambda_n = ((3e8/np.sqrt(6)) / f[0,n])
            h_whole = np.zeros((num_points_y, num_points_x), dtype=complex)
            for idx, (x_s, y_s) in enumerate(source_locations):
                y_idx = idx // num_points_x
                x_idx = idx % num_points_x
                h = 0
                for receive in range(200):
                    lambda_n = ((3e8/np.sqrt(6)) / f[0,n])
                    r_g = np.sqrt((da[0, receive] - 0)**2 + (0 - 0)**2)
                    x = F[n, receive].conjugate() 
                    r_s = np.sqrt((x_s - (da[0, receive] - 2.13))**2 + (y_s - 0)**2)
                    h_n = x * np.exp(-1j * 2 * np.pi * (2 * r_s) / lambda_n)
                    h += h_n
                h_whole[y_idx, x_idx] = h
            complex_wave.append(h_whole)
        return complex_wave


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="choose time domain or frequency domain")
    parser.add_argument('--part_A', action='store_true', help='different frequency')
    parser.add_argument('--part_B', action='store_true', help='different receivers')
    args = parser.parse_args()

    complex_wave = calculate()  
    np.save('F:/desktop/results/complex_wave.npy', complex_wave)