
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple

def chebyshev_nodes(n:int=10):
    if isinstance(n, int):
        A = np.array([])
        for k in range(n + 1):
            A = np.append(A, np.cos(k * np.pi / n))
        return A
    else:
        return None

    
def bar_czeb_weights(n:int=10):
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n, int):
        czeb_weights = np.zeros(n + 1)
        for j in range(n + 1):
            if j == 0 or j == n:
                gamma_j = 1/2
                czeb_weights[j] = (-1)**j * gamma_j
            else:
                gamma_j = 1
                czeb_weights[j] = (-1)**j * gamma_j
        return czeb_weights
    else: 
        return None
        

def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not all(isinstance(i, np.ndarray) for i in [xi, yi, wi, x]):
        return None
    elif xi.shape != yi.shape or yi.shape != wi.shape:
        return None
    else:
        Y = np.array([])
        for i in x:
            if i in xi:
                Y = np.append(Y, yi[np.where(xi == i)[0][0]])
            else:
                L = wi/(i-xi)
                Y = np.append(Y, yi @ L / sum(L))
        return Y

    
def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if all(isinstance(i, (int, float)) for i in [xr, x]):
        return np.abs(xr - x)
    elif all(isinstance(i, np.ndarray) for i in [xr, x]):
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN
    elif all(isinstance(i, list) for i in [xr, x]):
        return np.abs(max(xr) - max(x))
    else:
        return np.NaN
     
