import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse
from tqdm import tqdm

def generate_receivers_locations(lambda_0, y_offset, span):
    delta_x = lambda_0 / 4
    num_points = int(span / delta_x) + 1
    x_positions = np.linspace(-span / 2, span / 2, num_points)
    scatter_locations = [(x, y_offset) for x in x_positions]
    # print(scatter_locations)
    return scatter_locations
    
def generate_source_region(lambda_0, X_length, Y_length):
    delta_x = lambda_0 / 4
    num_points_x = int(X_length / delta_x) + 1
    num_points_y = int(Y_length / delta_x) + 1 
    x_positions = np.linspace(-X_length / 2, X_length / 2, num_points_x)
    y_positions = np.linspace(-Y_length / 2, Y_length / 2, num_points_y)
    source_locations = [(x, y) for y in y_positions for x in x_positions]
    print(source_locations)
    return source_locations

def calculate(source_scatter):

    lambda_0 = 1
    X_length = 60 * lambda_0
    Y_length = 60 * lambda_0
    y_offset = 60 * lambda_0
    span = 60 * lambda_0
    source_locations = generate_source_region(lambda_0, X_length, Y_length)
    receivers_locations = generate_receivers_locations(lambda_0, y_offset, span)

    num_points_x = int(X_length / (lambda_0 / 4)) + 1
    num_points_y = int(Y_length / (lambda_0 / 4)) + 1
    h_whole = np.zeros((num_points_y, num_points_x), dtype=complex)
    each_scatter = np.zeros((num_points_y, num_points_x), dtype=complex)
    temp = np.zeros((num_points_y, num_points_x), dtype=complex)
    complex_wave = []
    num = 0
    for x_g, y_g in tqdm((receivers_locations)):
        for n in range(64):
            lambda_n = 64*lambda_0/(32+n)
            for x_cc, y_cc in source_scatter:
                for idx, (x_s, y_s) in enumerate(source_locations):
                    y_idx = idx // num_points_x
                    x_idx = idx % num_points_x
                    r_g = np.sqrt((x_g - x_cc)**2 + (y_g - y_cc)**2)
                    r_s = np.sqrt((x_s - x_g)**2 + (y_s - y_g)**2)
                    h = np.exp(1j * 2 * np.pi * (r_g) / lambda_n) * np.exp(-1j * 2 * np.pi * (r_s) / lambda_n)
                    h_whole[y_idx, x_idx] = h
                each_scatter += h_whole
                h_whole = np.zeros((num_points_y, num_points_x), dtype=complex)
            temp += each_scatter
            each_scatter = np.zeros((num_points_y, num_points_x), dtype=complex)
        np.save(f'F:/desktop/results/complex_wave_{num}.npy', temp)
        num += 1
        complex_wave.append(temp)
        temp = np.zeros((num_points_y, num_points_x), dtype=complex)
    return complex_wave


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Different question in the assignment')
    args = parser.parse_args()
    source_scatter = [(0, 15), (-12, -9), (12, -9)]

    complex_wave = calculate(source_scatter)  
    np.save('F:/desktop/results/complex_wave.npy', complex_wave)
