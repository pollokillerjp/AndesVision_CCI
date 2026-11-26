"""Parte G · Simulador de la cola FIFO con c GPUs (programado sin librerías de eventos discretos)
y decisión de capacidad (Figura 9)."""
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, special

from .estadistica import ic_media_t

C0 = 2                        # GPUs del pool actual
CANDIDATOS_C = range(1, 6)    # números de GPUs evaluados en G2 y G3
WARMUP = 180                  # calentamiento descartado en cada réplica (s)


# =============================================================================
# Ajuste de los parámetros del simulador
# =============================================================================
def mle_gamma(x):
    """EMV de Gamma(k, θ): resuelve log k − ψ(k) = log(media) − media(log x). Devuelve (k, θ)."""
    x = np.asarray(x, float)
    s = np.log(x.mean()) - np.log(x).mean()
    k = optimize.brentq(lambda k: np.log(k) - special.digamma(k) - s, 1e-3, 1e4)
    return k, x.mean() / k


def mle_lognormal(x):
    """EMV de Lognormal: media y desviación (ddof = 0) de log x."""
    lx = np.log(np.asarray(x, float))
    return lx.mean(), lx.std(ddof=0)


def mle_gamma_condicional(x, medias):
    """EMV de la forma κ de G | tam ~ Gamma(κ, media_i/κ) con las medias lineales fijas."""
    x, m = np.asarray(x, float), np.asarray(medias, float)
    nll = lambda lk: -np.sum(stats.gamma.logpdf(x, np.exp(lk), scale=m / np.exp(lk)))
    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))


def ajustar(d, dur_estable, n_llegadas=None, cortes_tam=None):
    """Parámetros del simulador a partir de solicitudes del régimen estable.
    Modelo marginal: h, Gamma para G y Lognormal para la red.
    Modelo estructural (partes C y D): tam → P(caché | tam) → G | tam ~ Gamma con media lineal; red = loc + Gamma.
    `cortes_tam` son los deciles de tamaño que definen P(caché | tam); si no se pasan, se calculan con `d`."""
    n = len(d) if n_llegadas is None else n_llegadas
    h = d["cache_hit"].dropna().mean()
    gnc = d.loc[d["cache_hit"] == 0, "gpu_ms"].dropna()
    k, th = mle_gamma(gnc)
    mu, sg = mle_lognormal(d["red_ms"].dropna())
    # --- modelo estructural ---
    dd = d.dropna(subset=["tam_kb", "cache_hit"])
    if cortes_tam is None:
        cortes_tam = np.quantile(dd.tam_kb, np.linspace(0, 1, 11)[1:-1])
    p_cache = dd.groupby(np.digitize(dd.tam_kb, cortes_tam)).cache_hit.mean().reindex(range(10)).fillna(h).to_numpy()
    nc = d[(d.cache_hit == 0)].dropna(subset=["tam_kb", "gpu_ms"])
    X = np.column_stack([np.ones(len(nc)), nc.tam_kb])
    a, b = np.linalg.solve(X.T @ X, X.T @ nc.gpu_ms.to_numpy())       # media lineal de G | tam
    kappa = mle_gamma_condicional(nc.gpu_ms, a + b * nc.tam_kb)
    kr, loc_r, th_r = stats.gamma.fit(d["red_ms"].dropna())             # red = loc + Gamma
    return dict(lam=n / dur_estable, h=h, k=k, theta=th, mu_red=mu, sd_red=sg,
                p_cache=p_cache, a=a, b=b, kappa=kappa, kr=kr, loc_r=loc_r, th_r=th_r,
                tam=dd.tam_kb.to_numpy(), cortes_tam=cortes_tam)


def muestras_g(d):
    """Arreglos de gpu_ms (no cacheadas) y red_ms observados, para el simulador empírico."""
    return {"gpu": d.loc[d.cache_hit.eq(0), "gpu_ms"].dropna().to_numpy(),
            "red": d.red_ms.dropna().to_numpy()}


# =============================================================================
# Simulador de la cola
# =============================================================================
def simular(c, lam, p, T, rng_rep, warmup=300.0, muestras=None, modelo="estructural"):
    """Una réplica independiente de la cola FIFO con c GPUs durante T segundos.
    Devuelve las latencias L (ms) de las solicitudes que llegan después del calentamiento.
    modelo='estructural' (partes C y D) o 'marginal' (h, Gamma y Lognormal sin tamaño).
    muestras: dict con arreglos 'gpu' y 'red' para remuestrear datos reales en lugar del modelo."""
    n = rng_rep.poisson(lam * T)
    llegadas = np.sort(rng_rep.uniform(0.0, T, n))             # Poisson homogéneo condicionado a n llegadas
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
    libres = [0.0] * c                                         # montículo con el instante (s) en que cada GPU queda libre
    idx = np.flatnonzero(~cache)                               # solo las solicitudes no cacheadas usan GPU
    llg, srv = llegadas[idx], gpu[idx] / 1000.0
    esp = np.empty(len(idx))
    for j in range(len(idx)):                                  # cola FIFO con c servidores
        primera = libres[0]                                    # GPU que se libera antes
        # El servicio empieza en el mayor de dos instantes: la llegada y la liberación de esa GPU.
        inicio = llg[j] if llg[j] > primera else primera
        esp[j] = inicio - llg[j]
        heapq.heapreplace(libres, inicio + srv[j])             # la GPU queda ocupada hasta el fin del servicio
    espera[idx] = esp * 1000.0
    L = np.where(cache, red + rng_rep.uniform(1.0, 3.0, n), red + espera + gpu)
    return L[llegadas >= warmup]


def replicar(c, p, muestras, cantidad, horizonte, modelo_g, P, rng):
    """Proporción de latencias sobre τ en `cantidad` réplicas independientes con la tasa proyectada λ·g.
    Cada réplica recibe su propio generador hijo de `rng`."""
    tasa_proyectada = p["lam"] * P.g                 # solicitudes por segundo
    marcas = muestras if modelo_g == "empírico" else None
    proporciones = []
    for rng_replica in rng.spawn(cantidad):
        latencias_ms = simular(c, tasa_proyectada, p, horizonte, rng_replica, warmup=WARMUP, muestras=marcas)
        proporciones.append(np.mean(latencias_ms > P.tau))
    return np.asarray(proporciones)


def mmc(c, param_g, EG_nc, red_media, P):
    """P(L > τ) bajo una cola M/M/c (fórmula de Erlang C) con servicio exponencial y red constante."""
    mu = 1000 / EG_nc                                      # tasa de servicio por GPU (1/s)
    lam_nc = P.g * param_g["lam"] * (1 - param_g["h"])     # llegadas que van a GPU (1/s)
    a = lam_nc / mu                                        # carga ofrecida (erlangs)
    if a >= c:
        return np.nan
    cola = a**c / special.factorial(c) * c / (c - a)
    C = cola / (sum(a**k / special.factorial(k) for k in range(c)) + cola)   # probabilidad de esperar
    beta = c * mu - lam_nc                                 # la espera, dado que se espera, es Exp(beta)
    tt = (P.tau - red_media) / 1000                        # tiempo disponible para espera + servicio (s)
    suma = np.exp(-mu*tt) * (1 + mu*tt) if abs(beta - mu) < 1e-9 else (beta*np.exp(-mu*tt) - mu*np.exp(-beta*tt)) / (beta - mu)
    return (1 - param_g["h"]) * ((1 - C) * np.exp(-mu * tt) + C * suma)


# =============================================================================
# G0 · Ajuste · G1 · Validación · G2 · Capacidad · G3 · Incertidumbre
# =============================================================================
def g0_ajustar(df_estable, dur_estable):
    """Ajusta los parámetros del simulador con el régimen estable y calcula la carga por GPU con el pool actual."""
    param_g = ajustar(df_estable, dur_estable)
    EG_nc = param_g["k"] * param_g["theta"]                       # E[G | no caché] (ms) con la Gamma ajustada
    rho_base = param_g["lam"] * (1 - param_g["h"]) * EG_nc / 1000 / C0
    print("Parámetros del simulador:", {k: (round(float(v), 5) if np.isscalar(v) else "array")
                                        for k, v in param_g.items()})
    print(f"E[G | no caché] = {EG_nc:.2f} ms;  carga por GPU con c=2: rho = {rho_base:.3f}")

    evid = df_estable.assign(espera=df_estable.latencia_ms - df_estable.red_ms - df_estable.gpu_ms) \
                     .groupby("intentos")[["gpu_ms", "latencia_ms", "espera"]].mean().round(2)
    evid["n"] = df_estable["intentos"].value_counts().sort_index()
    print("Evidencia sobre reintentos (medias por número de intentos):\n", evid)
    return {"param_g": param_g, "EG_nc": EG_nc, "rho_base": rho_base, "evid": evid}


def g1_validacion(df_estable, G0, P, rng, rep_g1=30):
    """Compara los cuantiles de latencia reales con los simulados con 2 GPUs (modelo estructural y remuestreo empírico)
    y elige para la proyección el modelo con menor error absoluto en P(L > τ)."""
    param_g = G0["param_g"]
    reales = df_estable.latencia_ms.dropna().to_numpy()
    mues_g = muestras_g(df_estable)
    g1 = {"real": reales}
    for tipo in ("estructural", "empírico"):
        g1[tipo] = np.concatenate([simular(2, param_g["lam"], param_g, 1800, rng_rep, warmup=WARMUP,
                                           muestras=mues_g if tipo == "empírico" else None)
                                   for rng_rep in rng.spawn(rep_g1)])     # una semilla hija por réplica
    valid_g = pd.DataFrame({k: [*np.quantile(v, [.5, .9, .95, .99, .999]), np.mean(v > P.tau)]
                            for k, v in g1.items()}, index=["p50", "p90", "p95", "p99", "p99.9", "P(L>tau)"])
    e_real = valid_g.loc["P(L>tau)", "real"]
    dist = {k: abs(valid_g.loc["P(L>tau)", k] - e_real) for k in ("estructural", "empírico")}
    modelo_g = min(dist, key=dist.get)
    print("Modelo para proyección por error absoluto de cola G1:", modelo_g, dist)
    return {"g1": g1, "valid_g": valid_g, "modelo_g": modelo_g, "mues_g": mues_g}


def figura_g1(G1, ruta_g):
    """Figura de validación: cola P(L > x) observada y simulada con 2 GPUs."""
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
    """P(L > τ) con tráfico × g para c = 1, …, 5 GPUs con IC t sobre réplicas independientes.
    c* es el menor número de GPUs cuyo límite superior del IC cumple el SLO."""
    param_g, EG_nc = G0["param_g"], G0["EG_nc"]
    filas_g = []
    for c in CANDIDATOS_C:
        rho = P.g * param_g["lam"] * (1 - param_g["h"]) * EG_nc / (1000 * c)
        # Con ρ ≥ 1 las GPUs no alcanzan a vaciar la cola y no existe régimen estacionario.
        if rho >= 1:
            filas_g.append((c, rho, np.nan, np.nan, np.nan, 0))
            continue
        # Un horizonte de 2700 s deja 42 min útiles tras el calentamiento.
        pro = replicar(c, param_g, G1["mues_g"], replicas, horizonte, G1["modelo_g"], P, rng)
        mg, lo_g, hi_g = ic_media_t(pro, P.conf)
        filas_g.append((c, rho, mg, max(0, lo_g), min(1, hi_g), len(pro)))
    g2 = pd.DataFrame(filas_g, columns=["GPUs", "rho", "p", "lo", "hi", "R"]).set_index("GPUs")
    g2.to_csv(ruta_g / "G2_escenario_grupo.csv")
    c_g = next(iter(g2.index[g2.hi.le(P.eps)]), None)
    print("c* por extremo superior:", c_g)
    return {"g2": g2, "c_g": c_g}


def figura_9(G2, P, ruta_g):
    """Figura 9: P(L > τ) con su IC frente al número de GPUs y la línea del SLO."""
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
    """Distribución de c* por bootstrap: se remuestrean las filas para reajustar las marcas y el número de llegadas
    se sortea de una Poisson para reflejar la incertidumbre de λ. Agrega a la tabla de G2 la referencia M/M/c."""
    param_g, EG_nc, g2 = G0["param_g"], G0["EG_nc"], G2["g2"]
    conteo_c = {}
    for _ in range(B):
        muestra = df_estable.iloc[rng.integers(0, len(df_estable), len(df_estable))]
        pb_g = ajustar(muestra, dur_estable, rng.poisson(len(df_estable)), cortes_tam=param_g["cortes_tam"])
        marcas_b = muestras_g(muestra)
        numero = None
        for c in CANDIDATOS_C:
            if P.g * pb_g["lam"] * (1 - pb_g["h"]) * pb_g["k"] * pb_g["theta"] / (1000 * c) >= 1:
                continue                                   # ρ ≥ 1: capacidad insuficiente
            prop_b = replicar(c, pb_g, marcas_b, 8, 1800, G1["modelo_g"], P, rng)
            if prop_b.mean() <= P.eps:
                numero = c
                break
        clave = str(numero) if numero is not None else ">5"
        conteo_c[clave] = conteo_c.get(clave, 0) + 1
    print(f"Distribución de c* bootstrap ({B} muestras):", conteo_c)

    red_media = df_estable.red_ms.mean()
    g2["p_MMc"] = [mmc(c, param_g, EG_nc, red_media, P) for c in g2.index]
    print("M/M/c impone servicio exponencial y red constante. La cola empírica preserva la distribución observada de servicio, sin modelar cambios bajo saturación.")
    return {"conteo_c": conteo_c, "g2": g2}
