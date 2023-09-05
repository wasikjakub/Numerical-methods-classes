import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) == False or isinstance(b, (int, float)) == False or isinstance(epsilon, float) == False or isinstance(iteration, int) == False:
        return None
    if f(a) * f(b) <= 0: 
        for i in range(iteration):
            c = (b+a)/2
            if np.abs(f(c)) <= epsilon:
                return c, i
            if f(a)*f(c) < 0:
                a = a
                b = c
            elif f(c)*f(a) > 0:
                b = b
                a = c
            else:
                return c, iteration
    else:
        return None


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(iteration, int) and epsilon > 0 and iteration > 0:
        if f(a) * f(b) < 0:
            for i in range(iteration):
                x = (f(b)*a - f(a)*b)/(f(b)-f(a))
                if f(a)*f(x) <= 0:
                    b = x
                else:
                    a = x
                if abs(b-a) < epsilon:
                    return x, i
                elif abs(f(x)) < epsilon:
                    return x, i
            return(f(b) * a - f(a) * b) / (f(b) - f(a)), iteration
    else:
        return None

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(f, typing.Callable) and isinstance(df, typing.Callable) and isinstance(ddf, typing.Callable) and isinstance(a, (int, float)) and isinstance(b, (int, float))and isinstance(epsilon, float) and isinstance(iteration, int) and epsilon > 0 and iteration >= 0:
        if f(a) * f(b) < 0:
            if b > a:
                x = np.linspace(a, b, 1000)
                if df(a) > 0:
                    for i in x:
                        if df(i) <= 0:
                            return None
                else:
                    for i in x:
                        if df(i) >= 0:
                            return None
                if ddf(a) > 0:
                    for i in x:
                        if ddf(i) <= 0:
                            return None
                else:
                    for i in x:
                        if ddf(i) >= 0:
                            return None  
                for j in  x:
                    if df(j) == 0 or ddf(j) == 0:
                        return None
                for i in range(iteration + 1):
                    a = a - f(a) / df(a)
                    if np.abs(f(a)) < epsilon:
                        return a, i - 1                     
                return None
    else:
        return None

