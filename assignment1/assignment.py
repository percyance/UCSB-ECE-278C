import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import argparse

def question(lambda_0, scatter_locations, num):
    radius_limit = 60 * lambda_0
    N = 512  

    delta_x = lambda_0 / 4
    x_max = delta_x * N / 2
    x = np.linspace(-x_max, x_max, N)
    y = np.linspace(-x_max, x_max, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    r[r == 0] = 1
    h = np.zeros((N, N), dtype=complex) 
    for lambda_, x_n, y_n in scatter_locations:
        r_n = np.sqrt((X - x_n)**2 + (Y - y_n)**2)
        r_n[r_n == 0] = np.finfo(float).eps 
        if num == 4:
            h_n = 1 * np.exp(1j * 2 * np.pi * r_n / lambda_)
            h = h + h_n
        else:
            h_n = (1j * lambda_ * r_n)**(-0.5) * np.exp(1j * 2 * np.pi * r_n / lambda_)
            h = h + h_n

    h[r > radius_limit] = 0

    H = fftshift(fft2(h))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(h), extent=(x.min(), x.max(), y.min(), y.max()))
    plt.title("2D Coherent Wavefield Pattern")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(H), extent=(x.min(), x.max(), y.min(), y.max()))
    plt.title("Amplitude of 2D Fourier Spectrum")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Different question in the assignment')
    parser.add_argument('--Q1', action='store_true', help='Process question 1')
    parser.add_argument('--Q2', action='store_true', help='Process question 2')
    parser.add_argument('--Q3', action='store_true', help='Process question 3')
    parser.add_argument('--Q4', action='store_true', help='Process question 4')

    args = parser.parse_args()

    if args.Q1:
        lambda_0 = 1
        scatter_locations = [(lambda_0, 0, 0)]
        question(lambda_0, scatter_locations, 0)
    elif args.Q2:
        lambda_0 = lambda_1 = lambda_2 = 1
        scatter_locations = [(lambda_0, 0, 15 * lambda_0), (lambda_1, -12 * lambda_0, -9 * lambda_0), (lambda_2, 12 * lambda_0, -9 * lambda_0)]
        question(lambda_0, scatter_locations, 0)
    elif args.Q3:
        lambda_0 = 1
        lambda_1 = 0.5
        lambda_2 = 2
        scatter_locations = [(lambda_0, 0, 15 * lambda_0), (lambda_1, -12 * lambda_0, -9 * lambda_0), (lambda_2, 12 * lambda_0, -9 * lambda_0)]
        question(lambda_0, scatter_locations, 0)
    elif args.Q4:
        lambda_0 = 1
        scatter_locations = [(lambda_0, 0, 0)]
        question(lambda_0, scatter_locations, 4)


