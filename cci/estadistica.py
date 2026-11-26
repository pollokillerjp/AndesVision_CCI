"""Intervalos de confianza y otras funciones que usan varias partes."""
import numpy as np
from scipy import stats


def ic_wald(x, n, conf):
    """IC de Wald para una proporción. Devuelve (p, inferior, superior)."""
    zz = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    se = np.sqrt(p * (1 - p) / n)
    return p, p - zz * se, p + zz * se


def ic_wilson(x, n, conf):
    """IC de Wilson para una proporción (funciona mejor que Wald con p o n pequeños)."""
    zz = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    centro = (p + zz**2 / (2 * n)) / (1 + zz**2 / n)
    semi = zz * np.sqrt(p * (1 - p) / n + zz**2 / (4 * n**2)) / (1 + zz**2 / n)
    return p, centro - semi, centro + semi


def ic_dif_newcombe(x1, n1, x2, n2, conf):
    """IC de Newcombe para p2 - p1, armado con los intervalos de Wilson de cada grupo."""
    p1, l1, u1 = ic_wilson(x1, n1, conf)
    p2, l2, u2 = ic_wilson(x2, n2, conf)
    d = p2 - p1
    return d, d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def ic_media_t(x, conf):
    """IC t para la media (supone datos independientes)."""
    x = np.asarray(x, float)
    n = x.size
    m, s = x.mean(), x.std(ddof=1)
    t = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    return m, m - t * s / np.sqrt(n), m + t * s / np.sqrt(n)


def ic_dif_welch(xA, xB, conf):
    """IC de Welch para media(B) - media(A) con varianzas distintas."""
    xA, xB = np.asarray(xA, float), np.asarray(xB, float)
    vA, vB = xA.var(ddof=1) / xA.size, xB.var(ddof=1) / xB.size
    gl = (vA + vB) ** 2 / (vA**2 / (xA.size - 1) + vB**2 / (xB.size - 1))
    d = xB.mean() - xA.mean()
    t = stats.t.ppf(1 - (1 - conf) / 2, gl)
    return d, d - t * np.sqrt(vA + vB), d + t * np.sqrt(vA + vB)


def hessiano_numerico(f, x0, paso=1e-4):
    """Hessiano por diferencias centrales. Su inversa da la covarianza aproximada del EMV."""
    x0 = np.asarray(x0, float)
    d = len(x0)
    H = np.zeros((d, d))
    h = paso * np.maximum(np.abs(x0), 1.0)
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d)
            ej = np.zeros(d)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h[i] * h[j])
    return H


def ks_estadistico(x, cdf):
    """Estadístico D de Kolmogorov-Smirnov entre los datos x y la CDF del modelo."""
    ordenados = np.sort(x)
    n = len(ordenados)
    acumulada_modelo = cdf(ordenados)
    # la empírica salta en cada dato, por eso se mira antes y después del salto
    empirica_derecha = np.arange(1, n + 1) / n
    empirica_izquierda = np.arange(n) / n
    exceso_empirico = (empirica_derecha - acumulada_modelo).max()
    exceso_modelo = (acumulada_modelo - empirica_izquierda).max()
    return max(exceso_empirico, exceso_modelo)


def txt_ic(est, lo, hi, dec=3):
    return f"{est:.{dec}f}  [{lo:.{dec}f}, {hi:.{dec}f}]"


def a_nativo(o):
    """Pasa los tipos de numpy a tipos normales de Python para poder guardar en JSON."""
    if isinstance(o, dict):
        return {str(k): a_nativo(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [a_nativo(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o
