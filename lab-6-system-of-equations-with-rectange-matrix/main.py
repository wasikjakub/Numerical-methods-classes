import numpy as np
from numpy.core.function_base import linspace
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle

from typing import Union, List, Tuple


def spare_matrix_Abt(m: int,n: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (n,)  i pomocniczego wektora t (m,) zawierających losowe wartości
    Parameters:
    m(int): ilość wierszy macierzy A
    n(int): ilość kolumn macierzy A
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) == int and type(n) == int:
        t = np.linspace(0, 1, m)
        b = np.cos(4*t)
        A = np.vander(t, n)
        A = np.fliplr(A)
        return A, b
    else:
        return None

def square_from_rectan(A: np.ndarray, b: np.ndarray):
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników na kwadratowy układ równań. Funkcja ma zwrócić nową macierz współczynników  i nowy wektor współczynników
    Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (n,n) i wektorem (n,)
             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
     """
    if type(A) == np.ndarray and type(b) == np.ndarray:
        A_1 = np.transpose(A)
        lewa = A_1 @ A
        prawa = A_1 @ b
        return lewa, prawa
    else:
        return None



def residual_norm(A:np.ndarray,x:np.ndarray, b:np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b
      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania 
      x: wektor x (m.) zawierający rozwiązania równania 
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    if type(A) == np.ndarray and type(x) == np.ndarray and type(b) == np.ndarray:
            Ax = A @ np.transpose(x)
            res = b - Ax
            return np.linalg.norm(res)
    else:
            return None