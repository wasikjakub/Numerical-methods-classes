import numpy as np
import pickle
from random import randint

from typing import Union, List, Tuple

def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) == int and m >= 1:
        A = np.random.randint(0, 100,(m,m))*10
        b = np.random.randint(0, 100,(m,))*10
        return A, b
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
    if all(isinstance(i, (np.ndarray)) for i in [A, x, b]):
        if x.shape == b.shape and A.shape[1] == A.shape[0]:
            Ax = A @ np.transpose(x)
            res = b - Ax
            return np.linalg.norm(res)
        else:
            return None
    else:
        return None


def log_sing_value(n:int, min_order:Union[int,float], max_order:Union[int,float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """
    if all(isinstance(i, (int, float)) for i in [min_order, max_order]) and type(n) == int:
        if max_order > min_order and n > 0:
            return np.logspace(min_order, max_order, n)
        else:
            return None
    else:
        return None
    
def order_sing_value(n:int, order:Union[int,float] = 2, site:str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if type(n) == int and type(site) == str and isinstance(order, (int, float)):  
        if n > 0:
            wektor = np.random.rand(n)*10
            if site == "gre":
                wieksze = np.argmax(wektor)
                wektor[wieksze] += 10 ** order
            elif site == 'low':
                mniejsze = np.argmin(wektor)
                wektor[mniejsze] -= 10 ** order
            else:
                return None
            sorted_sing = np.sort(wektor)
            return np.flip(sorted_sing)
    return None


def create_matrix_from_A(A:np.ndarray, sing_value:np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    if all(isinstance(i, (np.ndarray)) for i in [A, sing_value]):
            if A.shape[0] == sing_value.shape[0] and A.shape[1] == A.shape[0]:
                u, s, vh = np.linalg.svd(A, full_matrices=True)
                return np.dot(u * sing_value, vh)
    return None