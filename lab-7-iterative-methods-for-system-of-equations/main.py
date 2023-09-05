import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) == int and m > 0:
        A = np.random.randint(0, 9, (m,m))
        b = np.random.randint(0, 9, (m,))
        A = A + np.diag([np.sum(A[i, :]) + np.sum(A[:, i]) for i in range(m)])
        return A, b
    else:    
        return None

def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(A) == np.ndarray and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        return all((2 * np.abs(np.diag(A))) >= sum(np.abs(A), 1))
    else:
        return None


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) == int and m > 0:
        a = np.random.randint(0, 9, (m,m))
        b = np.random.randint(0, 9, (m,))
        m = np.tril(a) + np.tril(a, -1).T
        return m, b
    else:
        return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(A) == np.ndarray and len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        return np.allclose(A, A.T)
    else:
        return None



def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if type(A) == np.ndarray and type(b) == np.ndarray and type(x_init) == np.ndarray and type(epsilon) == float and type(maxiter) == int:
        if maxiter > 0 and epsilon > 0 and A.shape[0] == A.shape[1] and b.shape[0] == A.shape[1] and b.shape[0] == x_init.shape[0]:
            k = 0
            D = np.diag(np.diag(A))
            LU = A - D
            x = x_init
            D_inv = np.diag(1 / np.diag(D))
            for i in range(maxiter):
                k += 1
                x_new = np.dot (D_inv, b - np.dot(LU, x))
                r_norm = np.linalg. norm(x_new - x)
                if r_norm<epsilon:
                    return x_new, k
                x = x_new
            return x, maxiter
    else:
        return None
    
## funkcja z laboratorium 6

def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(m) == int and m >= 1:
        A = np.random.randint(1, 100,(m,m), dtype = int)
        b = np.random.randint(1, 100,(m,), dtype = int)
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
            res = b - A@x
            return np.linalg.norm(res)
        else:
            return None
    else:
        return None