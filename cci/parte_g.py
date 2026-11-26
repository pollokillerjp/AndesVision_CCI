"""Parte G: simulador de la cola con c GPUs (hecho a mano, sin librerías de eventos discretos)
y decisión de capacidad (Figura 9)."""
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, special

from .estadistica import ic_media_t

C0 = 2                        # GPUs que hay hoy
CANDIDATOS_C = range(1, 6)    # cantidades de GPUs que se prueban
WARMUP = 180                  # segundos iniciales que se descartan en cada réplica


def mle_gamma(x):
    """EMV de una Gamma resolviendo log(k) - digamma(k) = log(media) - media(log x)."""
    x = np.asarray(x, float)
    s = np.log(x.mean()) - np.log(x).mean()
    k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-3, 1e4)
    return k, x.mean() / k


def mle_lognormal(x):
    lx = np.log(np.asarray(x, float))
    return lx.mean(), lx.std(ddof=0)


def mle_gamma_condicional(x, medias):
    """EMV de la forma kappa cuando cada G tiene su propia media (ya conocida)."""
    x, m = np.asarray(x, float), np.asarray(medias, float)
    nll = lambda lk: -np.sum(stats.gamma.logpdf(x, np.exp(lk), scale=m / np.exp(lk)))
    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))


def ajustar(d, dur_estable, n_llegadas=None, cortes_tam=None):
    """Parámetros del simulador a partir de los datos del régimen estable.
    Modelo estructural: tamaño -> P(caché | tamaño) -> G | tamaño Gamma con media lineal, y red = loc + Gamma.
    También se guardan los parámetros del modelo marginal (h, Gamma, Lognormal).
    cortes_tam son los deciles de tamaño; si no se dan se calculan con d."""
    n = len(d) if n_llegadas is None else n_llegadas
    h = d["cache_hit"].dropna().mean()
    gnc = d.loc[d["cache_hit"] == 0, "gpu_ms"].dropna()
    k, th = mle_gamma(gnc)
    mu, sg = mle_lognormal(d["red_ms"].dropna())

    dd = d.dropna(subset=["tam_kb", "cache_hit"])
    if cortes_tam is None:
        cortes_tam = np.quantile(dd.tam_kb, np.linspace(0, 1, 11)[1:-1])
    p_cache = dd.groupby(np.digitize(dd.tam_kb, cortes_tam)).cache_hit.mean().reindex(range(10)).fillna(h).to_numpy()
    nc = d[(d.cache_hit == 0)].dropna(subset=["tam_kb", "gpu_ms"])
    X = np.column_stack([np.ones(len(nc)), nc.tam_kb])
    a, b = np.linalg.solve(X.T @ X, X.T @ nc.gpu_ms.to_numpy())       # recta de la media de G según el tamaño
    kappa = mle_gamma_condicional(nc.gpu_ms, a + b * nc.tam_kb)
    kr, loc_r, th_r = stats.gamma.fit(d["red_ms"].dropna())
    return dict(lam=n / dur_estable, h=h, k=k, theta=th, mu_red=mu, sd_red=sg,
                p_cache=p_cache, a=a, b=b, kappa=kappa, kr=kr, loc_r=loc_r, th_r=th_r,
                tam=dd.tam_kb.to_numpy(), cortes_tam=cortes_tam)


def muestras_g(d):
    return {"gpu": d.loc[d.cache_hit.eq(0), "gpu_ms"].dropna().to_numpy(),
            "red": d.red_ms.dropna().to_numpy()}


def simular(c, lam, p, T, rng_rep, warmup=300.0, muestras=None, modelo="estructural"):
    """Simula la cola FIFO con c GPUs durante T segundos y devuelve las latencias (ms)
    de las solicitudes que llegan después del calentamiento.
    Si se pasa `muestras`, G y la red se toman de los datos reales en vez del modelo."""
    n = rng_rep.poisson(lam * T)
    llegadas = np.sort(rng_rep.uniform(0.0, T, n))      # dado n, las llegadas Poisson son uniformes
    if muestras is None and modelo == "estructural":
        tam = rng_rep.choice(p["tam"], n)
        cache = rng_rep.random(n) < p["p_cache"][np.digitize(tam, p["cortes_tam"])]
        gpu = rng_rep.gamma(p["kappa"], (p["a"] + p["b"] * tam) / p["kappa"])
        red = p["loc_r"] + rng_rep.gamma(p["kr"], p["th_r"], n)
    elif muestras is None:
        cache = rng_rep.random(n) < p["h"]
        red = rng_rep.lognormal(p["mu_red"], p["sd_red"], n)
        gpu = rng_rep.gamma(p["k"], p["theta"], n)
    else:
        cache = rng_rep.random(n) < p["h"]
        red = rng_rep.choice(muestras["red"], n)
        gpu = rng_rep.choice(muestras["gpu"], n)

    espera = np.zeros(n)
    libres = [0.0] * c                 # heap con el momento en que se desocupa cada GPU
    idx = np.flatnonzero(~cache)       # las respuestas de caché no pasan por la cola
    llg, srv = llegadas[idx], gpu[idx] / 1000.0
    esp = np.empty(len(idx))
    for j in range(len(idx)):
        primera = libres[0]                                   # la GPU que se libera primero
        inicio = llg[j] if llg[j] > primera else primera      # empieza cuando llega o cuando hay GPU libre
        esp[j] = inicio - llg[j]
        heapq.heapreplace(libres, inicio + srv[j])            # esa GPU queda ocupada hasta inicio + servicio
    espera[idx] = esp * 1000.0
    L = np.where(cache, red + rng_rep.uniform(1.0, 3.0, n), red + espera + gpu)
    return L[llegadas >= warmup]


def replicar(c, p, muestras, cantidad, horizonte, modelo_g, P, rng):
    """Proporción de latencias mayores a tau en varias réplicas con el tráfico multiplicado por g."""
    tasa_proyectada = p["lam"] * P.g
    marcas = muestras if modelo_g == "empírico" else None
    proporciones = []
    for rng_replica in rng.spawn(cantidad):      # cada réplica con su propio generador
        latencias_ms = simular(c, tasa_proyectada, p, horizonte, rng_replica, warmup=WARMUP, muestras=marcas)
        proporciones.append(np.mean(latencias_ms > P.tau))
    return np.asarray(proporciones)


def mmc(c, param_g, EG_nc, red_media, P):
    """P(L > tau) con una cola M/M/c (Erlang C), suponiendo servicio exponencial y red constante."""
    mu = 1000 / EG_nc                                      # servicios por segundo de una GPU
    lam_nc = P.g * param_g["lam"] * (1 - param_g["h"])     # llegadas que van a la GPU
    a = lam_nc / mu
    if a >= c:
        return np.nan
    cola = a**c / special.factorial(c) * c / (c - a)
    C = cola / (sum(a**k / special.factorial(k) for k in range(c)) + cola)   # probabilidad de tener que esperar
    beta = c * mu - lam_nc                                 # si espera, la espera es exponencial con tasa beta
    tt = (P.tau - red_media) / 1000                        # tiempo que queda para espera + servicio
    if abs(beta - mu) < 1e-9:
        suma = np.exp(-mu * tt) * (1 + mu * tt)
    else:
        suma = (beta * np.exp(-mu * tt) - mu * np.exp(-beta * tt)) / (beta - mu)
    return (1 - param_g["h"]) * ((1 - C) * np.exp(-mu * tt) + C * suma)


def g0_ajustar(df_estable, dur_estable):
    param_g = ajustar(df_estable, dur_estable)
    EG_nc = param_g["k"] * param_g["theta"]                       # E[G | no caché]
    rho_base = param_g["lam"] * (1 - param_g["h"]) * EG_nc / 1000 / C0
    # medias por número de intentos (para ver si los reintentos cambian el servicio)
    evid = df_estable.assign(espera=df_estable.latencia_ms - df_estable.red_ms - df_estable.gpu_ms) \
                     .groupby("intentos")[["gpu_ms", "latencia_ms", "espera"]].mean().round(2)
    evid["n"] = df_estable["intentos"].value_counts().sort_index()
    print(f"lambda = {param_g['lam']:.3f} sol/s, h = {param_g['h']:.4f}, E[G | no caché] = {EG_nc:.2f} ms, "
          f"carga por GPU con {C0} GPUs = {rho_base:.3f}")
    return {"param_g": param_g, "EG_nc": EG_nc, "rho_base": rho_base, "evid": evid}


def g1_validacion(df_estable, G0, P, rng, rep_g1=30):
    """Con 2 GPUs compara cuantiles reales y simulados (modelo estructural y remuestreo de datos)
    y se queda con el que mejor reproduce P(L > tau)."""
    param_g = G0["param_g"]
    reales = df_estable.latencia_ms.dropna().to_numpy()
    mues_g = muestras_g(df_estable)
    g1 = {"real": reales}
    for tipo in ("estructural", "empírico"):
        g1[tipo] = np.concatenate([simular(2, param_g["lam"], param_g, 1800, rng_rep, warmup=WARMUP,
                                           muestras=mues_g if tipo == "empírico" else None)
                                   for rng_rep in rng.spawn(rep_g1)])
    valid_g = pd.DataFrame({k: [*np.quantile(v, [.5, .9, .95, .99, .999]), np.mean(v > P.tau)]
                            for k, v in g1.items()}, index=["p50", "p90", "p95", "p99", "p99.9", "P(L>tau)"])
    e_real = valid_g.loc["P(L>tau)", "real"]
    dist = {k: abs(valid_g.loc["P(L>tau)", k] - e_real) for k in ("estructural", "empírico")}
    modelo_g = min(dist, key=dist.get)
    print(f"Error en P(L>tau): estructural = {dist['estructural']:.5f}, empírico = {dist['empírico']:.5f}. "
          f"Se usa el modelo {modelo_g} para proyectar.")
    return {"g1": g1, "valid_g": valid_g, "modelo_g": modelo_g, "mues_g": mues_g}


def figura_g1(G1, ruta_g):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in G1["g1"].items():
        xx = np.linspace(15, 320, 250)
        ax.plot(xx, 1 - np.searchsorted(np.sort(v), xx, side="right") / len(v), label=k)
    ax.set(xlabel="Latencia (ms)", ylabel="P(L > x)", yscale="log", title="G1 · Cola observada y simulada con 2 GPUs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_g / "figuraG1_validacion.png", dpi=150)
    plt.show()


def g2_capacidad(G0, G1, P, rng, ruta_g, replicas=40, horizonte=2700):
    """P(L > tau) con el tráfico por g para 1 a 5 GPUs. c* es el menor c cuyo límite superior del IC cumple el SLO."""
    param_g, EG_nc = G0["param_g"], G0["EG_nc"]
    filas_g = []
    for c in CANDIDATOS_C:
        rho = P.g * param_g["lam"] * (1 - param_g["h"]) * EG_nc / (1000 * c)
        if rho >= 1:              # la cola crece sin límite, no tiene sentido simular
            filas_g.append((c, rho, np.nan, np.nan, np.nan, 0))
            continue
        # 2700 s menos 180 de calentamiento dejan 42 minutos útiles por réplica
        proporciones = replicar(c, param_g, G1["mues_g"], replicas, horizonte, G1["modelo_g"], P, rng)
        p_media, p_inf, p_sup = ic_media_t(proporciones, P.conf)
        filas_g.append((c, rho, p_media, max(0, p_inf), min(1, p_sup), len(proporciones)))
    g2 = pd.DataFrame(filas_g, columns=["GPUs", "rho", "p", "lo", "hi", "R"]).set_index("GPUs")
    g2.to_csv(ruta_g / "G2_escenario_grupo.csv")
    c_g = next(iter(g2.index[g2.hi.le(P.eps)]), None)
    print(f"c* = {c_g} GPUs (menor c con límite superior del IC <= {P.eps})")
    return {"g2": g2, "c_g": c_g}


def figura_9(G2, P, ruta_g):
    g2 = G2["g2"]
    fig, ax = plt.subplots(figsize=(6, 4))
    g2_ok = g2.dropna(subset=["p"])
    ax.errorbar(g2_ok.index, 100 * g2_ok.p, yerr=100 * np.array([g2_ok.p - g2_ok.lo, g2_ok.hi - g2_ok.p]),
                fmt="o-", capsize=4, label=f"IC t {P.conf:.0%}")
    ax.axhline(100 * P.eps, color="firebrick", ls="--", label="SLO")
    ax.set(xlabel="Número de GPUs", ylabel=f"P(L > {P.tau} ms) (%)", xticks=list(CANDIDATOS_C),
           title=f"Figura 9 · Tráfico × {P.g:.1f}")
    for c in g2.index[g2.R.eq(0)]:
        ax.text(c, 100 * P.eps, "ρ ≥ 1", ha="center", va="bottom")
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_g / "figura9_c_vs_P.png", dpi=160)
    plt.show()


def g3_incertidumbre(df_estable, dur_estable, G0, G1, G2, P, rng, B=100):
    """Bootstrap de c*: se remuestrean las filas, se reajustan los parámetros y el número de llegadas
    se saca de una Poisson. Al final se agrega la columna con la aproximación M/M/c."""
    param_g, EG_nc, g2 = G0["param_g"], G0["EG_nc"], G2["g2"]
    conteo_c = {}
    for _ in range(B):
        muestra = df_estable.iloc[rng.integers(0, len(df_estable), len(df_estable))]
        param_boot = ajustar(muestra, dur_estable, rng.poisson(len(df_estable)), cortes_tam=param_g["cortes_tam"])
        muestras_boot = muestras_g(muestra)
        c_min = None
        for c in CANDIDATOS_C:
            if P.g * param_boot["lam"] * (1 - param_boot["h"]) * param_boot["k"] * param_boot["theta"] / (1000 * c) >= 1:
                continue
            prop_boot = replicar(c, param_boot, muestras_boot, 8, 1800, G1["modelo_g"], P, rng)
            if prop_boot.mean() <= P.eps:
                c_min = c
                break
        clave = str(c_min) if c_min is not None else ">5"
        conteo_c[clave] = conteo_c.get(clave, 0) + 1

    red_media = df_estable.red_ms.mean()
    g2["p_MMc"] = [mmc(c, param_g, EG_nc, red_media, P) for c in g2.index]
    print(f"c* en {B} muestras bootstrap: {conteo_c}")
    return {"conteo_c": conteo_c, "g2": g2}
