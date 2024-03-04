import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse

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
    # print(source_locations)
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
    temp = np.zeros((num_points_y, num_points_x), dtype=complex)
    source_scatter = [(0, 0)]

    for x_cc, y_cc in source_scatter:
        for idx, (x_s, y_s) in enumerate(source_locations):
            y_idx = idx // num_points_x
            x_idx = idx % num_points_x
            h = 0
            for x_g, y_g in receivers_locations:
                r_g = np.sqrt((x_g - x_cc)**2 + (y_g - y_cc)**2)
                r_s = np.sqrt((x_s - x_g)**2 + (y_s - y_g)**2)
                h_n = (1/(lambda_0**2 * r_g * r_s)) * np.exp(1j * 2 * np.pi * (r_g - r_s) / lambda_0)
                h += h_n
            h_whole[y_idx, x_idx] = h
        temp += h_whole
    return temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Different question in the assignment')
    parser.add_argument('--Q1', action='store_true', help='Process one target')
    parser.add_argument('--Q2', action='store_true', help='Process three target')

    args = parser.parse_args()
    if args.Q1:
        source_scatter = [(0, 0)]
    elif args.Q2:
        source_scatter = [(0, 15), (-12, -9), (12, -9)]

    h_whole = calculate(source_scatter)

    # 绘制 h_whole 的实部和虚部
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(h_whole.real, extent=(-30, 30, -30, 30))
    plt.title('Real Part of h_whole')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(h_whole.imag, extent=(-30, 30, -30, 30))
    plt.title('Imaginary Part of h_whole')
    plt.colorbar()
    plt.show()

    # FFT 变换
    fft_image = fftshift(fft2(h_whole))

    # 绘制 FFT 幅度分布图
    plt.figure(figsize=(6, 5))
    plt.imshow(np.abs(fft_image), extent=(-30, 30, -30, 30))
    plt.title('FFT Magnitude Spectrum')
    plt.colorbar()
    plt.show()

    # 绘制重建图像的幅度分布
    plt.figure(figsize=(6, 5))
    plt.imshow(np.abs(h_whole), extent=(-30, 30, -30, 30))
    plt.title('Magnitude Distribution of Reconstructed Image')
    plt.colorbar()
    plt.show()
