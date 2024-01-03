import numpy as np
import matplotlib.pyplot as plt

SMOOTHING_LENGTH = 1

def gauss_kernel(radius : float):
    """
    Compute the Gaussian kernel.
    """
    a = 1 / (np.pi * SMOOTHING_LENGTH**2)
    b = radius/SMOOTHING_LENGTH
    k = 2

    if b <= k*SMOOTHING_LENGTH:
        return a * np.exp(-b**2)
    else:
        return 0.0001

    
def derivative_gaussian_kernel(radius : float):
    """
    Compute the derivative of Gaussian kernel.
    """
    a = 1 / (np.pi * SMOOTHING_LENGTH**2)
    b = radius/SMOOTHING_LENGTH
    k = 2

    if b <= k*SMOOTHING_LENGTH:
        return -2/(SMOOTHING_LENGTH**2) * b * a * np.exp(-b**2)
    else:
        return 0
    

def cubic_kernel(radius : float):
    """
    Compute the cubic spline kernel.
    """
    a = 10/ (7 * np.pi * SMOOTHING_LENGTH**2)
    k = radius/SMOOTHING_LENGTH

    if 0 <= k and k <= 1:
        return a * (1 - 1.5 * k**2 + 0.75 * k**3)
    elif 1 <= k and k <= 2:
        return a * 0.25 * (2 - k)**3
    else:
        return 0.001
    
def derivative_cubic_kernel(radius : float):
    """
    Compute the derivative of cubic spline kernel.
    """
    a = 10/ (7 * np.pi * SMOOTHING_LENGTH**2)
    k = radius/SMOOTHING_LENGTH

    if 0 <= k and k <= 1:
        return a * (-3/SMOOTHING_LENGTH * k + 2.25/SMOOTHING_LENGTH**2 * k**2)
    elif 1 <= k and k <= 2:
        return a * (-0.75 * (2 - k)**2)
    else:
        return 0
    
if __name__ == '__main__':
    # plot the kernel function
    x = np.linspace(0, 3, 100)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = cubic_kernel(x[i])
    plt.plot(x, y)
    plt.show()
