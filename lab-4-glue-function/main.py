from typing import List, Union
import numpy as np
import scipy
import pickle

#from typing import Union, List, Tuple


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.
    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów
    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    if type(x) != np.ndarray:
        return None
    if type(y) != np.ndarray:
        return None
    if x.shape != y.shape:
        return None
    a = np.array([])
    b = np.array([])
    for i in range(len(x) - 1):
        a = np.append(a, (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        b = np.append(b, y[i] - a[i] * x[i])

    return a, b


def cubic_spline(x, y, tol = 1e-100):
    if type(x) != np.ndarray:
        return None
    if type(y) != np.ndarray:
        return None
    if x.shape != y.shape:
        return None

    x = np.array(x)
    y = np.array(y)
    
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    
   
    A = np.zeros(shape = (size,size))
    b = np.zeros(shape=(size,1))
    A[0,0] = 1
    A[-1,-1] = 1
    
    for i in range(1,size-1):
        A[i, i-1] = delta_x[i-1]
        A[i, i+1] = delta_x[i]
        A[i,i] = 2*(delta_x[i-1]+delta_x[i])
    
        b[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])
        
    
    print('Jacobi Method Output:')
    c = jacobi(A, b, np.zeros(len(A)), tol = tol, n_iterations=1000)
    
   
    d = np.zeros(shape = (size-1,1))
    b = np.zeros(shape = (size-1,1))
    for i in range(0,len(d)):
        d[i] = (c[i+1] - c[i]) / (3*delta_x[i])
        b[i] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + c[i+1])    
    
    return b.squeeze(), c.squeeze(), d.squeeze()

def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.
    
    Returns:
    x, the estimated solution
    """
    
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol+1
    
    while (x_diff > tol) and (counter < n_iterations): #iteration level
        for i in range(0, n): #element wise level for x
            s = 0
            for j in range(0,n): #summation for i !=j
                if i != j:
                    s += A[i,j] * x_prev[j] 
            
            x[i] = (b[i] - s) / A[i,i]
        #update values
        counter += 1
        x_diff = (np.sum((x-x_prev)**2))**0.5 
        x_prev = x.copy() #use new x for next iteration
        
    
    print("Number of Iterations: ", counter)
    print("Norm of Difference: ", x_diff)
    return x

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
     
