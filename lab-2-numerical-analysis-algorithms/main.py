import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if type(v) == list:
        v = np.array(v)
    elif type(v_aprox) == list:
        v_aprox = np.array(v_aprox)
    elif type(v)==np.ndarray:
        if len(v) != len(v_aprox):
            return np.NaN
        return abs(v - v_aprox)
    else:
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if type(v) == int:
        if v == 0:
            return float('NaN')
    if not isinstance(v, (int, float, List, np.ndarray)) or not isinstance(v_aprox, (int, float, List, np.ndarray)):
            return float('NaN')
    if type(v) == list:
        v = np.array(v)
        if not all([e for e in v]):
            return float
    if type(v_aprox) == list:
        v_aprox = np.array(v_aprox)
    if type(v)==np.ndarray:
        if len(v) != len(v_aprox):
            return float('NaN')
    return abs((v - v_aprox) / v)



def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    b = 2**n
    P1=b-b+c
    P2=b+c-b
    return absolut_error(P1,P2)
   


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(x, (int, float)) or not isinstance(n, int) or n < 0:
        return float("NaN")
    sum = 0
    for n in range(n):
        sum += (x**n)/scipy.math.factorial(n)
    return sum



def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    metoda_1 = 2*np.cos(x)*coskx1((k-1),x) - coskx1((k-2), x)

    if not isinstance(x, (int, float)) or not isinstance(k, int):
        return np.nan
    if k == 1:
        return np.cos(x)
    if k == 0:
        return np.cos(0*x)
    else:
        return metoda_1

def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    metoda_2 = np.cos(x) * coskx2((k - 1), x)[0] - np.sin(x) * coskx2((k - 1), x)[1]
    metoda_2_2 = np.sin(x) * coskx2((k - 1), x)[0] + np.cos(x) * coskx2((k - 1), x)[1]

    if not isinstance(x, (int, float)) or not isinstance(k, int):
        return np.nan
    if k == 1:
        return np.cos(x), np.sin(x)
    if k == 0:
        return np.cos(0*x), np.sin(0*x)
    else:
        return metoda_2, metoda_2_2


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(n, int):
        return float("NaN")
    if n < 1:
        return float("NaN")
    s = 0
    for i in range(1, n + 1):
        s += 1 / i**2
    pi_aprox = np.sqrt(6 * s)
    return pi_aprox
        