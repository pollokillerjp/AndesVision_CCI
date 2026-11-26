"""Parte C · Tiempo de cómputo en GPU y tiempo de red (Figuras 2 y 3)."""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, special

from .estadistica import ic_wilson, ic_media_t, hessiano_numerico, ks_estadistico, txt_ic

B_KS = 200       # réplicas del bootstrap paramétrico del KS (C2)
R_MC = 100_000   # réplicas de la simulación por remuestreo (C4)


# =============================================================================
# Funciones auxiliares: log-verosimilitudes y ajuste por máxima verosimilitud
# =============================================================================
def nll_gamma(par, x):
    """−log-verosimilitud de Gamma(k, θ) para una muestra positiva x."""
    k, th = par
    if k <= 0 or th <= 0:
        return np.inf
    n = len(x)
    return -((k - 1) * np.log(x).sum() - x.sum() / th - n * k * np.log(th) - n * special.gammaln(k))


def nll_lognorm(par, x):
    """−log-verosimilitud de Lognormal(μ, s) para una muestra positiva x."""
    mu, s = par
    if s <= 0:
        return np.inf
    lx = np.log(x)
    return -(-lx.sum() - len(x) * np.log(s * np.sqrt(2 * np.pi)) - ((lx - mu) ** 2).sum() / (2 * s**2))


def nll_gamma_cond(par, x, y):
    """−log-verosimilitud de G | tam ~ Gamma(κ, media β0 + β1·tam): forma común y media lineal en el tamaño."""
    b0, b1, kap = par
    mu = b0 + b1 * x
    if kap <= 0 or np.any(mu <= 0):
        return np.inf
    return -(len(y) * (kap * np.log(kap) - special.gammaln(kap)) + ((kap - 1) * np.log(y) - kap * np.log(mu) - kap * y / mu).sum())


def nll_gamma_desp(par, x):
    """−log-verosimilitud de la Gamma desplazada x = loc + Gamma(k, θ), con loc < min(x)."""
    loc, k, th = par
    if k <= 0 or th <= 0 or loc >= x.min():
        return np.inf
    exceso = x - loc
    return -((k - 1) * np.log(exceso).sum() - exceso.sum() / th - len(x) * (k * np.log(th) + special.gammaln(k)))


def ajustar_emv(nll, x, inicio, positivos):
    """Minimiza la −log-verosimilitud `nll` con Nelder-Mead.
    Los parámetros marcados en `positivos` se optimizan en escala logarítmica para que sean > 0.
    Devuelve (estimaciones, errores estándar por hessiano observado, log-verosimilitud máxima)."""
    positivos = np.asarray(positivos)
    def a_natural(u):
        u = np.array(u, float); u[positivos] = np.exp(u[positivos]); return u
    u0 = np.array(inicio, float); u0[positivos] = np.log(u0[positivos])
    res = optimize.minimize(lambda u: nll(a_natural(u), x), u0, method="Nelder-Mead",
                            options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
    est = a_natural(res.x)
    H = hessiano_numerico(lambda p: nll(p, x), est)
    ee = np.sqrt(np.diag(np.linalg.inv(H)))
    return est, ee, -res.fun


def momentos_G(h, mu_c, var_c):
    """E[G] y Var(G) de la variable mixta: masa h en 0 y parte continua con media mu_c y varianza var_c."""
    return (1 - h) * mu_c, (1 - h) * var_c + h * (1 - h) * mu_c**2


def bloques_reales(datos, n_bloque):
    """Sumas de gpu_ms en bloques de `n_bloque` solicitudes no cacheadas consecutivas,
    formados dentro de cada tramo estable y en orden de llegada."""
    sumas = []
    for _, seg in datos.sort_values("t_llegada_s").groupby("segmento"):
        v = seg.loc[seg.cache_hit == 0, "gpu_ms"].dropna().to_numpy()
        nb = len(v) // n_bloque
        sumas.append(v[:nb * n_bloque].reshape(nb, n_bloque).sum(axis=1))
    return np.concatenate(sumas)


# =============================================================================
# C1 · Naturaleza de G (variable mixta)
# =============================================================================
def c1_masa_en_cero(df_estable, P, params):
    """Masa en cero h = P(caché) con IC de Wilson y parte continua G_c de las solicitudes no cacheadas.
    Usa solo las filas con caché y GPU válidos."""
    validas = df_estable.dropna(subset=["cache_hit", "gpu_ms"])
    assert ((validas.gpu_ms == 0) == (validas.cache_hit == 1)).all()   # G = 0 exactamente cuando hay caché
    G_todas = validas.gpu_ms.to_numpy()
    G_c = validas.loc[validas.cache_hit == 0, "gpu_ms"].to_numpy()
    n_validas, n_cache = len(validas), int((validas.cache_hit == 1).sum())
    h_hat, h_lo, h_hi = ic_wilson(n_cache, n_validas, P.conf)
    print(f"n = {n_validas} | caché = {n_cache}")
    print(f"h = P(G = 0) = {txt_ic(h_hat, h_lo, h_hi, 4)}  (Wilson, {P.conf:.0%})")
    print(f"Parte continua: n = {len(G_c)}, media = {G_c.mean():.3f} ms, varianza = {G_c.var(ddof=1):.2f}, "
          f"asimetría = {stats.skew(G_c):.3f}, curtosis = {stats.kurtosis(G_c):.3f}")
    params["h"] = {"est": h_hat, "ic": [h_lo, h_hi], "metodo": "Wilson"}
    return {"validas": validas, "G_todas": G_todas, "G_c": G_c, "h_hat": h_hat, "h_lo": h_lo, "h_hi": h_hi}


def c1_momentos(c1, c2, P):
    """Compara E[G] y Var(G) de la muestra con las fórmulas de la variable mixta (Gamma EMV y de momentos).
    La varianza total suma la dispersión de los tiempos positivos y el salto entre usar caché y usar GPU."""
    G_todas, h_hat = c1["G_todas"], c1["h_hat"]
    E_mv, V_mv = momentos_G(h_hat, c2["k_mv"] * c2["th_mv"], c2["k_mv"] * c2["th_mv"]**2)
    E_mm, V_mm = momentos_G(h_hat, c2["k_mm"] * c2["th_mm"], c2["k_mm"] * c2["th_mm"]**2)
    EG, EG_lo, EG_hi = ic_media_t(G_todas, P.conf)
    tabla = pd.DataFrame({
        "E[G] (ms)": [EG, E_mv, E_mm],
        "Var(G) (ms²)": [G_todas.var(ddof=1), V_mv, V_mm],
    }, index=[f"Muestra (IC {P.conf:.0%} de la media: {EG_lo:.2f} a {EG_hi:.2f})", "Fórmula con Gamma EMV", "Fórmula con Gamma momentos"])
    return {"E_mv": E_mv, "V_mv": V_mv, "E_mm": E_mm, "V_mm": V_mm, "EG": EG, "EG_lo": EG_lo, "EG_hi": EG_hi, "tabla_C1": tabla}


# =============================================================================
# C2 · Máxima verosimilitud para la parte continua
# =============================================================================
def c2_emv(G_c, P):
    """EMV propio de Gamma y Lognormal para G_c, con IC por hessiano, comparado con el método de momentos."""
    m_c, v_c = G_c.mean(), G_c.var(ddof=1)
    # Método de momentos: iguala la media y la varianza muestrales con las del modelo.
    k_mm, th_mm = m_c**2 / v_c, v_c / m_c
    s_mm = np.sqrt(np.log(1 + v_c / m_c**2)); mu_mm = np.log(m_c) - s_mm**2 / 2

    # Las estimaciones de momentos sirven como punto de partida del optimizador.
    (k_mv, th_mv), ee_gam, ll_gam = ajustar_emv(nll_gamma, G_c, [k_mm, th_mm], [True, True])
    (mu_mv, s_mv), ee_ln, ll_ln = ajustar_emv(nll_lognorm, G_c, [mu_mm, s_mm], [False, True])

    z = P.z_conf
    tabla_C2 = pd.DataFrame({
        "parámetro": ["Gamma k", "Gamma θ (ms)", "Lognormal μ", "Lognormal s"],
        "EMV": [k_mv, th_mv, mu_mv, s_mv],
        f"IC {P.conf:.0%} inf": [k_mv - z * ee_gam[0], th_mv - z * ee_gam[1], mu_mv - z * ee_ln[0], s_mv - z * ee_ln[1]],
        f"IC {P.conf:.0%} sup": [k_mv + z * ee_gam[0], th_mv + z * ee_gam[1], mu_mv + z * ee_ln[0], s_mv + z * ee_ln[1]],
        "Momentos": [k_mm, th_mm, mu_mm, s_mm],
    })
    print("Verificación (solo para comprobar el EMV propio): scipy.gamma.fit =", np.round(stats.gamma.fit(G_c, floc=0)[::2], 4))
    print(f"AIC Gamma = {2*2 - 2*ll_gam:.1f} | AIC Lognormal = {2*2 - 2*ll_ln:.1f}  (menor es mejor)")
    return {"m_c": m_c, "v_c": v_c, "k_mm": k_mm, "th_mm": th_mm, "s_mm": s_mm, "mu_mm": mu_mm,
            "k_mv": k_mv, "th_mv": th_mv, "ee_gam": ee_gam, "ll_gam": ll_gam,
            "mu_mv": mu_mv, "s_mv": s_mv, "ee_ln": ee_ln, "ll_ln": ll_ln, "tabla_C2": tabla_C2}


def c2_bondad_ajuste(G_c, c2, rng):
    """Bondad de ajuste de la Gamma: KS con valor p por bootstrap paramétrico y chi-cuadrado con 20 clases."""
    k_mv, th_mv, s_mv, mu_mv = c2["k_mv"], c2["th_mv"], c2["s_mv"], c2["mu_mv"]
    cdf_gam = lambda x: stats.gamma.cdf(x, k_mv, scale=th_mv)
    cdf_ln = lambda x: stats.lognorm.cdf(x, s_mv, scale=np.exp(mu_mv))
    D_gam, D_ln = ks_estadistico(G_c, cdf_gam), ks_estadistico(G_c, cdf_ln)

    # Cada réplica bootstrap se vuelve a ajustar, igual que la muestra original.
    D_boot = np.empty(B_KS)
    for b in range(B_KS):
        xb = rng.gamma(k_mv, th_mv, size=len(G_c))
        (kb, thb), _, _ = ajustar_emv(nll_gamma, xb, [k_mv, th_mv], [True, True])
        D_boot[b] = ks_estadistico(xb, lambda x: stats.gamma.cdf(x, kb, scale=thb))
    p_ks_gam = (1 + (D_boot >= D_gam).sum()) / (B_KS + 1)

    # Chi-cuadrado con 20 clases equiprobables bajo la Gamma ajustada (gl = 20 − 1 − 2 parámetros).
    bordes_chi2 = stats.gamma.ppf(np.linspace(0, 1, 21), k_mv, scale=th_mv)
    obs = np.histogram(G_c, bins=bordes_chi2)[0]
    esp = np.full(20, len(G_c) / 20)
    chi2_gam = ((obs - esp) ** 2 / esp).sum()
    p_chi2_gam = stats.chi2.sf(chi2_gam, 20 - 1 - 2)

    print(f"KS Gamma: D = {D_gam:.4f}, p (bootstrap paramétrico, B={B_KS}) = {p_ks_gam:.3f}")
    print(f"KS Lognormal: D = {D_ln:.4f}")
    print(f"Chi-cuadrado Gamma (20 bins): {chi2_gam:.1f}, gl = 17, p = {p_chi2_gam:.4f}")
    print(f"Varianza: muestral = {c2['v_c']:.1f} | Gamma EMV = {k_mv*th_mv**2:.1f} | Gamma momentos = {c2['k_mm']*c2['th_mm']**2:.1f}")
    return {"D_gam": D_gam, "D_ln": D_ln, "D_boot": D_boot, "p_ks_gam": p_ks_gam, "chi2_gam": chi2_gam, "p_chi2_gam": p_chi2_gam}


def c2_gamma_condicional(validas, c2, ajuste, P, params):
    """EMV de G | tam ~ Gamma(κ, β0 + β1·tam) y comparación de la marginal que implica con una sola Gamma."""
    k_mv, th_mv, ee_gam = c2["k_mv"], c2["th_mv"], c2["ee_gam"]
    cond = validas[(validas.cache_hit == 0) & validas.tam_kb.notna()]
    x_c, y_c = cond.tam_kb.to_numpy(), cond.gpu_ms.to_numpy()

    u0 = np.array([np.mean(y_c), 0.0, np.log(k_mv)])     # κ se optimiza en escala logarítmica
    res_cond = optimize.minimize(lambda u: nll_gamma_cond([u[0], u[1], np.exp(u[2])], x_c, y_c), u0, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 40000})
    b0_g, b1_g, kap_g = res_cond.x[0], res_cond.x[1], np.exp(res_cond.x[2])
    H_cond = hessiano_numerico(lambda p: nll_gamma_cond(p, x_c, y_c), [b0_g, b1_g, kap_g])
    ee_cond = np.sqrt(np.diag(np.linalg.inv(H_cond)))

    # Marginal implicada por el modelo condicional: F(y) = promedio de las CDF Gamma de cada imagen.
    mu_i = b0_g + b1_g * x_c
    rej = np.sort(y_c)
    F_mezcla = np.array([stats.gamma.cdf(v, kap_g, scale=mu_i / kap_g).mean() for v in rej[::10]])
    F_emp = np.arange(1, len(rej) + 1)[::10] / len(rej)
    D_mezcla = np.abs(F_emp - F_mezcla).max()
    var_mezcla = (mu_i**2 / kap_g).mean() + mu_i.var()     # varianza total = E[Var(G|tam)] + Var(E[G|tam])

    print(f"Modelo condicional (EMV): β0 = {b0_g:.3f} ± {ee_cond[0]:.3f}, β1 = {b1_g:.5f} ± {ee_cond[1]:.5f} ms/KB, κ = {kap_g:.3f} ± {ee_cond[2]:.3f}")
    print(f"Coeficiente de variación implicado 1/√κ = {1/np.sqrt(kap_g):.3f}")
    print(f"Distancia KS a los datos: una sola Gamma = {ajuste['D_gam']:.4f} | mezcla condicional ≈ {D_mezcla:.4f}")
    print(f"Varianza: datos = {np.var(y_c, ddof=1):.1f} | una sola Gamma = {k_mv*th_mv**2:.1f} | mezcla condicional = {var_mezcla:.1f}")
    params["G_dist"] = "gamma"
    params["G_par"] = {"k": k_mv, "theta": th_mv, "ic_k": [k_mv - P.z_conf*ee_gam[0], k_mv + P.z_conf*ee_gam[0]],
                       "ic_theta": [th_mv - P.z_conf*ee_gam[1], th_mv + P.z_conf*ee_gam[1]], "metodo": "EMV propio"}
    params["G_cond"] = {"b0": b0_g, "b1": b1_g, "kappa": kap_g, "ee": list(ee_cond),
                        "uso": "G | tam, no caché ~ Gamma(kappa, scale=(b0+b1*tam)/kappa)"}
    return {"b0_g": b0_g, "b1_g": b1_g, "kap_g": kap_g, "ee_cond": ee_cond, "D_mezcla": D_mezcla, "var_mezcla": var_mezcla}


def figura_2(c1, c2, ruta):
    """Figura 2: CDF de G con el salto en 0, histograma de la parte continua con los ajustes y gráfico Q-Q."""
    G_todas, G_c, h_hat = c1["G_todas"], c1["G_c"], c1["h_hat"]
    k_mv, th_mv, s_mv, mu_mv = c2["k_mv"], c2["th_mv"], c2["s_mv"], c2["mu_mv"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.linspace(0, np.quantile(G_todas, 0.995), 600)
    ecdf_x = np.sort(G_todas); ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
    ax[0].step(ecdf_x, ecdf_y, where="post", color="tab:blue", label="CDF empírica")
    ax[0].plot(xs, h_hat + (1 - h_hat) * stats.gamma.cdf(xs, k_mv, scale=th_mv), "k--", label="Modelo mixto: h + (1-h)·Gamma")
    ax[0].annotate(f"salto de tamaño h = {h_hat:.3f}", xy=(0, h_hat), xytext=(60, 0.25), arrowprops=dict(arrowstyle="->"))
    ax[0].set(title="(a) CDF de G (todas las solicitudes)", xlabel="G = gpu_ms (ms)", ylabel="P(G ≤ x)")
    ax[1].hist(G_c, bins=80, density=True, color="lightgray", edgecolor="white", label="G sin caché (datos)")
    xs2 = np.linspace(0.5, np.quantile(G_c, 0.999), 600)
    ax[1].plot(xs2, stats.gamma.pdf(xs2, k_mv, scale=th_mv), "tab:red", label=f"Gamma EMV (k={k_mv:.2f}, θ={th_mv:.2f})")
    ax[1].plot(xs2, stats.lognorm.pdf(xs2, s_mv, scale=np.exp(mu_mv)), "tab:green", ls="--", label="Lognormal EMV")
    ax[1].set(title="(b) Parte continua G_c y ajustes", xlabel="gpu_ms (ms)", ylabel="Densidad (1/ms)")
    qs = np.linspace(0.005, 0.995, 199)
    ax[2].scatter(stats.gamma.ppf(qs, k_mv, scale=th_mv), np.quantile(G_c, qs), s=8, color="tab:red", label="Gamma")
    ax[2].scatter(stats.lognorm.ppf(qs, s_mv, scale=np.exp(mu_mv)), np.quantile(G_c, qs), s=8, color="tab:green", label="Lognormal")
    lim = [0, np.quantile(G_c, 0.995)]
    ax[2].plot(lim, lim, "k:", label="y = x")
    ax[2].set(title="(c) Gráfico Q-Q", xlabel="Cuantil teórico (ms)", ylabel="Cuantil observado (ms)")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Figura 2. G es una variable mixta: un salto en 0 (caché) más una parte continua asimétrica", y=1.02)
    fig.text(0.01, -0.06, "El salto de la CDF en 0 es la proporción de respuestas desde caché. La parte continua es asimétrica a la derecha: la Gamma la sigue "
             "en el centro, pero se separa en la cola alta (panel c),\nporque en realidad es una mezcla de imágenes de distintos tamaños.", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta / "figura2_G.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# C3 · Tiempo de red
# =============================================================================
def c3_normalidad(df_estable):
    """Forma de la distribución de red_ms y tres pruebas de normalidad."""
    red = df_estable.red_ms.dropna().to_numpy()
    n_red = len(red)
    m_r, s_r = red.mean(), red.std(ddof=1)
    print(f"n = {n_red} | media = {m_r:.3f} | desv = {s_r:.3f} | asimetría = {stats.skew(red):.3f} | curtosis exceso = {stats.kurtosis(red):.3f} | mínimo = {red.min():.2f}")
    print("Criterio 1, D'Agostino-Pearson (asimetría + curtosis):", stats.normaltest(red))
    print("Criterio 2, Jarque-Bera:", stats.jarque_bera(red))
    with warnings.catch_warnings():                       # silencia el aviso de cambio de API de SciPy en anderson()
        warnings.simplefilter("ignore", FutureWarning)
        ad = stats.anderson(red, dist="norm")
    print(f"Criterio 3, Anderson-Darling: A² = {ad.statistic:.2f} (valor crítico al 1 %: {ad.critical_values[-1]:.3f})")
    return {"red": red, "n_red": n_red, "m_r": m_r, "s_r": s_r}


def c3_gamma_desplazada(c3n, params):
    """EMV propio de la Gamma desplazada red = loc + Gamma(k, θ), que respeta el piso mínimo de red,
    comparada con la normal por AIC y KS."""
    red, m_r, s_r = c3n["red"], c3n["m_r"], c3n["s_r"]
    gam_r = stats.skew(red)
    # Punto de partida por momentos: la asimetría de una Gamma es 2/√k.
    k0 = 4 / gam_r**2; th0 = s_r / np.sqrt(k0); loc0 = min(m_r - k0 * th0, red.min() - 0.5)
    # Reparametrización: loc = min(red) − e^u0 garantiza loc < min(red); k y θ en escala logarítmica.
    u0 = [np.log(red.min() - loc0), np.log(k0), np.log(th0)]
    a_nat = lambda u: [red.min() - np.exp(u[0]), np.exp(u[1]), np.exp(u[2])]
    res_red = optimize.minimize(lambda u: nll_gamma_desp(a_nat(u), red), u0, method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 40000})
    loc_r, k_r, th_r = a_nat(res_red.x)
    H_r = hessiano_numerico(lambda p: nll_gamma_desp(p, red), [loc_r, k_r, th_r])
    cov_r = np.linalg.inv(H_r)
    ll_norm = stats.norm.logpdf(red, m_r, red.std(ddof=0)).sum()
    print(f"Gamma desplazada EMV: loc = {loc_r:.3f} ms, k = {k_r:.3f}, θ = {th_r:.3f} ms")
    print(f"AIC normal = {2*2 - 2*ll_norm:.1f} | AIC Gamma desplazada = {2*3 + 2*res_red.fun:.1f}")
    print(f"KS normal: D = {ks_estadistico(red, lambda x: stats.norm.cdf(x, m_r, s_r)):.4f} | "
          f"KS Gamma desplazada: D = {ks_estadistico(red, lambda x: stats.gamma.cdf(x - loc_r, k_r, scale=th_r)):.4f}")
    params["red_par"] = {"dist": "gamma desplazada", "loc": loc_r, "k": k_r, "theta": th_r, "metodo": "EMV propio"}
    return {"loc_r": loc_r, "k_r": k_r, "th_r": th_r, "cov_r": cov_r}


def c3_colas(c3n, c3g, P):
    """P(red > x) para tres valores de x y percentil 99.9 % con IC: empírico, normal y Gamma desplazada."""
    red, n_red, m_r, s_r = c3n["red"], c3n["n_red"], c3n["m_r"], c3n["s_r"]
    loc_r, k_r, th_r, cov_r = c3g["loc_r"], c3g["k_r"], c3g["th_r"], c3g["cov_r"]
    filas = []
    for x_ms in [25, 30, 35]:
        p, lo, hi = ic_wilson((red > x_ms).sum(), n_red, P.conf)
        filas.append({"x (ms)": x_ms, "empírica": p, f"IC {P.conf:.0%} Wilson": f"[{lo:.5f}, {hi:.5f}]",
                      "modelo normal": stats.norm.sf(x_ms, m_r, s_r),
                      "Gamma desplazada": stats.gamma.sf(x_ms - loc_r, k_r, scale=th_r)})
    tabla_C3 = pd.DataFrame(filas)

    # Percentil 99.9 %: empírico con IC exacto por estadísticos de orden; normal y Gamma desplazada con IC por método delta.
    q999 = 0.999
    orden = np.sort(red)
    # El número de datos por debajo del cuantil verdadero es Binomial(n, q): sus cuantiles dan las posiciones del IC.
    i_inf = int(stats.binom.ppf(P.alpha / 2, n_red, q999)); i_sup = int(stats.binom.ppf(1 - P.alpha / 2, n_red, q999)) + 1
    p999_emp = np.quantile(red, q999)
    zq = stats.norm.ppf(q999)
    p999_norm = m_r + zq * s_r
    ee_norm = s_r * np.sqrt(1 / n_red + zq**2 / (2 * n_red))
    f_q = lambda p: p[0] + stats.gamma.ppf(q999, p[1], scale=p[2])     # cuantil de la Gamma desplazada
    theta_r = np.array([loc_r, k_r, th_r])
    # Gradiente numérico del cuantil respecto a (loc, k, θ) para el método delta.
    grad = np.array([(f_q(theta_r + e) - f_q(theta_r - e)) / (2 * e[i]) for i, e in enumerate(np.eye(3) * 1e-5 * np.maximum(np.abs(theta_r), 1))])
    p999_gam = f_q(theta_r); ee_gam_q = np.sqrt(grad @ cov_r @ grad)
    ic_red = [orden[i_inf - 1], orden[min(i_sup, n_red) - 1]]
    z = P.z_conf
    print(tabla_C3.round(5).to_string(index=False))
    print(f"\nPercentil 99.9 % (IC {P.conf:.0%}):")
    print(f"  empírico (estadísticos de orden): {p999_emp:.2f} ms  [{ic_red[0]:.2f}, {ic_red[1]:.2f}]")
    print(f"  normal (método delta):            {txt_ic(p999_norm, p999_norm - z*ee_norm, p999_norm + z*ee_norm, 2)} ms")
    print(f"  Gamma desplazada (método delta):  {txt_ic(p999_gam, p999_gam - z*ee_gam_q, p999_gam + z*ee_gam_q, 2)} ms")
    return {"tabla_C3": tabla_C3, "p999_emp": p999_emp, "ic_red": ic_red, "p999_norm": p999_norm, "ee_norm": ee_norm,
            "p999_gam": p999_gam, "ee_gam_q": ee_gam_q}


def figura_3(c3n, c3g, ruta):
    """Figura 3: red_ms frente a la normal y a la Gamma desplazada (densidad, Q-Q y cola derecha)."""
    red, n_red, m_r, s_r = c3n["red"], c3n["n_red"], c3n["m_r"], c3n["s_r"]
    loc_r, k_r, th_r = c3g["loc_r"], c3g["k_r"], c3g["th_r"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.linspace(red.min() - 5, red.max(), 500)
    ax[0].hist(red, bins=80, density=True, color="lightgray", edgecolor="white", label="red_ms (datos)")
    ax[0].plot(xs, stats.norm.pdf(xs, m_r, s_r), "tab:blue", label="Normal")
    ax[0].plot(xs, stats.gamma.pdf(xs - loc_r, k_r, scale=th_r), "tab:red", label="Gamma desplazada")
    ax[0].set(title="(a) Histograma y densidades", xlabel="red_ms (ms)", ylabel="Densidad (1/ms)")
    qs = np.linspace(0.001, 0.999, 300)
    ax[1].scatter(stats.norm.ppf(qs, m_r, s_r), np.quantile(red, qs), s=6, color="tab:blue", label="vs. Normal")
    ax[1].scatter(loc_r + stats.gamma.ppf(qs, k_r, scale=th_r), np.quantile(red, qs), s=6, color="tab:red", label="vs. Gamma desplazada")
    ax[1].plot([0, 100], [0, 100], "k:", label="y = x")
    ax[1].set(title="(b) Gráfico Q-Q", xlabel="Cuantil teórico (ms)", ylabel="Cuantil observado (ms)", xlim=(0, 100), ylim=(0, 100))
    xs3 = np.linspace(30, 100, 300)
    ax[2].semilogy(np.sort(red), 1 - np.arange(1, n_red + 1) / (n_red + 1), color="black", label="Empírica")
    ax[2].semilogy(xs3, stats.norm.sf(xs3, m_r, s_r), "tab:blue", label="Normal")
    ax[2].semilogy(xs3, stats.gamma.sf(xs3 - loc_r, k_r, scale=th_r), "tab:red", label="Gamma desplazada")
    for x_ms in [50, 60, 80]:
        ax[2].axvline(x_ms, color="gray", ls=":", lw=1)
    ax[2].set(title="(c) Cola derecha: P(red > x)", xlabel="x (ms)", ylabel="P(red > x), escala log", xlim=(30, 100), ylim=(1e-5, 1))
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Figura 3. El tiempo de red no es normal: tiene un piso mínimo y una cola derecha más pesada", y=1.02)
    fig.text(0.01, -0.06, "La normal pone probabilidad por debajo del piso de ~10 ms y subestima la cola: con ella, P(red > 80 ms) sale miles de veces menor que lo observado.\n"
             "La Gamma desplazada sigue a los datos en todo el rango, así que es la que debe usar el simulador.", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta / "figura3_red.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# C4 · Suma de N tiempos de GPU
# =============================================================================
def c4_suma_gpu(G_c, df_estable, c2, P, rng):
    """P(T_N > N(m̂ + 15)) por TLC, Gamma, Edgeworth, remuestreo y bloques reales consecutivos,
    para la N del grupo y, como sensibilidad, para N entre 10 y 20."""
    m_hat, s_hat, gam_hat = G_c.mean(), G_c.std(ddof=1), stats.skew(G_c)
    k_mm, th_mm = c2["k_mm"], c2["th_mm"]

    def prob_C4(n_):
        """Probabilidad de que n_ solicitudes no cacheadas sumen más de n_(m̂ + 15) ms, por cada método."""
        umbral = n_ * (m_hat + 15)
        z = 15 * np.sqrt(n_) / s_hat                   # umbral estandarizado de la suma
        tlc = stats.norm.sf(z)
        gamma_exacta = stats.gamma.sf(umbral, n_ * k_mm, scale=th_mm)
        edgeworth = stats.norm.sf(z) + stats.norm.pdf(z) * gam_hat / (6 * np.sqrt(n_)) * (z**2 - 1)
        sim = (rng.choice(G_c, size=(R_MC, n_)).sum(axis=1) > umbral).mean()
        sb = bloques_reales(df_estable, n_)
        return {"N": n_, "TLC": tlc, "Gamma exacta": gamma_exacta, "Edgeworth": edgeworth,
                "Simulación (remuestreo)": sim, "Bloques reales": (sb > umbral).mean(), "n bloques": len(sb)}

    N = P.N
    res_N = prob_C4(N)
    sb = bloques_reales(df_estable, N)
    pb, pb_lo, pb_hi = ic_wilson((sb > N * (m_hat + 15)).sum(), len(sb), P.conf)
    ee_sim = np.sqrt(res_N["Simulación (remuestreo)"] * (1 - res_N["Simulación (remuestreo)"]) / R_MC)
    print(f"N = {N}, m̂ = {m_hat:.3f} ms, umbral = {N*(m_hat+15):.1f} ms")
    for metodo, valor in res_N.items():
        if metodo not in ("N", "n bloques"):
            print(f"  {metodo:<24} {valor:.5f}")
    print(f"  error estándar de la simulación: {ee_sim:.5f} | bloques reales: IC {P.conf:.0%} Wilson [{pb_lo:.4f}, {pb_hi:.4f}] con {len(sb)} bloques")

    # Sensibilidad a N: error relativo de cada aproximación frente a la simulación.
    tabla_C4 = pd.DataFrame([res_N if n_ == N else prob_C4(n_) for n_ in range(10, 21)]).set_index("N")
    for col in ["TLC", "Gamma exacta", "Edgeworth"]:
        tabla_C4[f"error {col} (%)"] = 100 * (tabla_C4[col] / tabla_C4["Simulación (remuestreo)"] - 1)
    print(f"\nError relativo medio frente a la simulación (N = 10 a 20): "
          f"TLC {tabla_C4['error TLC (%)'].abs().mean():.1f} % | Gamma {tabla_C4['error Gamma exacta (%)'].abs().mean():.1f} % | "
          f"Edgeworth {tabla_C4['error Edgeworth (%)'].abs().mean():.1f} %")
    print(f"Asimetría real de G = {gam_hat:.3f} | asimetría que impone la Gamma de momentos = 2/√k = {2/np.sqrt(k_mm):.3f}")
    return {"m_hat": m_hat, "s_hat": s_hat, "gam_hat": gam_hat, "res_N": res_N, "pb_lo": pb_lo, "pb_hi": pb_hi,
            "ee_sim": ee_sim, "tabla_C4": tabla_C4}
