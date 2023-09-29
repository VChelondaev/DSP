from scipy.integrate import quad
from scipy.signal import square
import numpy as np
import matplotlib.pyplot as plt

'''
def function(x):
    return square(x*3) * 2'''


def function(t):
    T = 2
    duty_cycle = 0.5
    A = 2
    return square(2 * np.pi * t / T, duty=duty_cycle) * A

def function_2(x):
    f = 1
    A = 4
    w = 2 * np.pi * f
    return A * np.cos(w * x)


def fourier_series(x, T, N, function):
    a0 = quad(function, 0, T)[0]
    a0 *= (2 / T)
    res = a0/2
    result = np.array([res] * len(x))
    for n in range(N):
        an = quad((lambda el: function(el) * np.cos(n * el * 2 * np.pi / T)), 0, T)[0]
        an *= (2 / T)
        bn = quad((lambda el: function(el) * np.sin(n * el * 2 * np.pi / T)), 0, T)[0]
        bn *= (2 / T)
        result += ((an * np.cos(n * x * 2 * np.pi / T)) + (bn * np.sin(n * x * 2 * np.pi / T)))
        eps = function(x) - result
    return result, eps, an

x = np.linspace(-4, 4, 1000)
x_2 = np.linspace(-1, 1, 1000)
T = 2
N = 10

y, eps, an = fourier_series(x, T, N, function)




'''plt.plot(x, y, label='Fourier Series Approximation')
plt.plot(x, function(x), label='Original Function')
plt.legend()
plt.show()

plt.plot(x, eps, label='error')
plt.legend()
plt.show()'''

res, eps, an = fourier_series(x_2, T, N, function_2)

squre_signal = square(x)



'''plt.plot(x_2, res, label='Fourier Series Approximation')
plt.plot(x_2, function_2(x_2), label='Original Function cos')
plt.legend()
plt.show()'''

t = np.fft.fft(res)

plt.plot(x_2*100, t, label='Спектральный коэффициент спектра')
plt.xlim(xmin=0)
plt.show()

t_sq = np.fft.fft(y)
plt.plot(x_2*100, t_sq, label='Спектральный коэффициент спектра')
plt.xlim(xmin=0)
plt.show()


squre_signal += np.random.normal(scale=30, loc=10)


t_sq = np.fft.fft(squre_signal)
fftfreq = np.fft.fftfreq(len(x), 1 / 1000)
plt.bar(fftfreq, np.fft.fft(squre_signal), label='Спектральный коэффициент спектра')
plt.show()