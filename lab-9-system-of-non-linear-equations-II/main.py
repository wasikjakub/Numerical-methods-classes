import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(x) == np.ndarray:
        x1 = P.polyfromroots(x)
        return x1
    else:
        return None

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(a) == np.ndarray:
        a1 = np.array(a)
        for i in range(20):
            a2 = np.random.random_sample(a.shape[0])*10e-10
            a1 = a1 + a2
            a3 = P.polyroots(a1)
        return a1, a3
    else:
        return None


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if type(wsp) == np.ndarray:
        x = np.eye(wsp.shape[0] - 1)
        x2 = np.zeros((wsp.shape[0] - 1 , 1))
        x = np.concatenate((x2,x), axis=1)
        x = np.concatenate((x, np.reshape(-wsp, (1, wsp.shape[0]))), axis = 0)
        x3 = np.linalg.eigvals(x)
        x4 = scipy.linalg.schur(x)
        x5 = P.polyroots(wsp)
        return x, x3 , x4 , x5
    else:
        return None
