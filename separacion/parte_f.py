"""Parte F · Inferencia, distribuciones muestrales y comparación de versiones (Figuras 7 y 8)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .estadistica import ic_wald, ic_wilson, ic_media_t, ic_dif_welch, ic_dif_newcombe
from .parte_a import limpiar

M_F1 = 10_000                 # muestras por cada tamaño n en F1
N_GRAFICA_F1 = [5, 30, 200, 2000]
N_BUSQUEDA_F1 = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
K_LOTES = 30                  # lotes de F2
R4 = 10_000                   # réplicas por configuración en F4
NS_F4 = [10, 30, 100, 500]
PS_F4 = [0.01, 0.02, 0.05, 0.10, 0.30, 0.50]
R_F5 = 2000                   # réplicas de la verificación de F5


# =============================================================================
# Funciones auxiliares
# =============================================================================
def autocorr(x, lag):
    """Autocorrelación muestral de x en el retardo `lag`."""
    x = x - x.mean()
    return np.sum(x[:-lag] * x[lag:]) / np.sum(x * x)


def dif_estandarizada(a, b):
    """Diferencia de medias estandarizada (SMD) entre dos grupos; |SMD| < 0.1 indica grupos comparables."""
    return (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)


def f6_datos(d, dur, tau):
    """Variables que F6 compara entre la ventana de referencia y la de validación."""
    gpu = d.loc[d.cache_hit.eq(0), "gpu_ms"].dropna()
    red = d.red_ms.dropna()
    lat = d.latencia_ms.dropna()
    intentos = d.intentos.dropna()
    cache_d = d.cache_hit.dropna()
    return {"tasa": len(d) / dur, "gpu": gpu, "red": red, "lat": lat, "reintentos": intentos.gt(1),
            "cache": cache_d.eq(1), "violaciones": lat.gt(tau)}


# =============================================================================
# F1 · Distribución muestral de la media
# =============================================================================
def f1_distribucion_muestral(df_estable, P, rng):
    """Forma de la distribución de x̄ de la latencia y cobertura efectiva del IC normal para varios n.
    La población es la distribución empírica de la latencia del régimen estable (muestreo con reemplazo)."""
    poblacion = df_estable["latencia_ms"].dropna().to_numpy()   # NaN = medición inválida (Parte A)
    mu_pob, sd_pob = poblacion.mean(), poblacion.std()
    print(f"Población: media={mu_pob:.2f} ms, sd={sd_pob:.2f}, "
          f"asimetría={stats.skew(poblacion):.2f}, curtosis exceso={stats.kurtosis(poblacion):.2f}")

    def medias_y_cobertura(n, M=M_F1, bloque=1000):
        """M medias muestrales de tamaño n y la fracción de IC normales que contienen mu_pob."""
        medias, cubre = [], 0
        for _ in range(M // bloque):                 # remuestreo por lotes para limitar la memoria
            muestras = rng.choice(poblacion, size=(bloque, n), replace=True)
            xb = muestras.mean(axis=1)
            s = muestras.std(axis=1, ddof=1)
            cubre += np.sum(np.abs(xb - mu_pob) <= P.z_conf * s / np.sqrt(n))
            medias.append(xb)
        return np.concatenate(medias), cubre / M

    tabla_f1 = []
    medias_por_n = {}
    for n in N_GRAFICA_F1:
        xb, cob = medias_y_cobertura(n)
        medias_por_n[n] = xb
        tabla_f1.append({"n": n, "sd(x̄)": xb.std(), "σ/√n": sd_pob / np.sqrt(n),
                         "asimetría": stats.skew(xb), "curtosis_exceso": stats.kurtosis(xb),
                         "cobertura": cob})
    tabla_f1 = pd.DataFrame(tabla_f1)
    print(tabla_f1.round(4))

    # Menor n con cobertura ≥ conf − 0.01
    objetivo = P.conf - 0.01
    n_min = None
    for n in N_BUSQUEDA_F1:
        _, cob = medias_y_cobertura(n)
        print(f"n={n:5d}  cobertura={cob:.4f}")
        if cob >= objetivo:
            n_min = n
            break
    print(f"Menor n con cobertura >= {objetivo:.2f}: n = {n_min}")
    return {"mu_pob": mu_pob, "sd_pob": sd_pob, "tabla_f1": tabla_f1, "medias_por_n": medias_por_n, "n_min": n_min}


def figura_7(F1, ruta):
    """Figura 7: distribución muestral de la media frente a la normal del TLC para cuatro tamaños de muestra."""
    medias_por_n, mu_pob, sd_pob = F1["medias_por_n"], F1["mu_pob"], F1["sd_pob"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, n in zip(axs, N_GRAFICA_F1):
        xb = medias_por_n[n]
        ax.hist(xb, bins=60, density=True, alpha=0.6, label="simulada")
        xs = np.linspace(xb.min(), xb.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu_pob, sd_pob / np.sqrt(n)), "r", label="Normal (TLC)")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("media muestral de latencia (ms)")
    axs[0].set_ylabel("densidad")
    axs[0].legend(fontsize=8)
    fig.suptitle("Figura 7. Distribución muestral de la media de latencia_ms")
    plt.figtext(0.01, -0.05, "Con n pequeño la media hereda la asimetría de la latencia; "
                "al crecer n se acerca a la normal y su dispersión cae como σ/√n.", fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta / "figura7_tlc.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# F2 · IC para la latencia media
# =============================================================================
def f2_ic_latencia(df_estable, P):
    """IC t clásico frente a IC por medias de lotes, que tolera la autocorrelación de latencias consecutivas.
    Los lotes se forman dentro de cada tramo estable."""
    lat_t = df_estable.sort_values("t_llegada_s")["latencia_ms"].dropna().to_numpy()
    m_cl, lo_cl, hi_cl = ic_media_t(lat_t, P.conf)

    acf = {k: autocorr(lat_t, k) for k in [1, 2, 5, 10, 50]}
    banda = P.z_conf / np.sqrt(lat_t.size)          # banda aproximada si no hubiera autocorrelación
    print("Autocorrelación:", {k: round(v, 3) for k, v in acf.items()}, f" banda ±{banda:.3f}")

    lotes = []
    for _, tramo in df_estable.sort_values("t_llegada_s").groupby("segmento"):
        lat_tramo = tramo["latencia_ms"].dropna().to_numpy()
        k_tramo = max(2, round(K_LOTES * len(lat_tramo) / len(lat_t)))   # lotes proporcionales al tamaño del tramo
        lotes += np.array_split(lat_tramo, k_tramo)
    medias_lote = np.array([b.mean() for b in lotes])
    print(f"Autocorrelación lag-1 entre medias de lote: {autocorr(medias_lote, 1):.3f}")
    m_bm, lo_bm, hi_bm = ic_media_t(medias_lote, P.conf)

    print(f"IC clásico  {P.conf:.0%}: {m_cl:.2f} [{lo_cl:.2f}, {hi_cl:.2f}]  ancho={hi_cl-lo_cl:.2f}")
    print(f"IC por lotes {P.conf:.0%}: {m_bm:.2f} [{lo_bm:.2f}, {hi_bm:.2f}]  ancho={hi_bm-lo_bm:.2f}")
    print(f"El IC corregido es {(hi_bm-lo_bm)/(hi_cl-lo_cl):.2f} veces más ancho")
    return {"m_cl": m_cl, "lo_cl": lo_cl, "hi_cl": hi_cl, "acf": acf, "medias_lote": medias_lote,
            "m_bm": m_bm, "lo_bm": lo_bm, "hi_bm": hi_bm}


# =============================================================================
# F3 · Versión B frente a A
# =============================================================================
def f3_comparar_versiones(df_estable, P, params):
    """Balance de covariables entre versiones y diferencias B − A en GPU media (Welch), P(intentos > 1)
    y P(L > τ) (Newcombe), con corrección de Bonferroni para confianza conjunta 1 − α."""
    df_A = df_estable[df_estable["version"] == "A"]
    df_B = df_estable[df_estable["version"] == "B"]

    print("Balance entre versiones (|dif. estandarizada| < 0.1 = comparables):")
    for col in ["tam_kb", "red_ms", "cache_hit"]:
        print(f"  {col:10s} SMD = {dif_estandarizada(df_A[col], df_B[col]):+.3f}")
    x_prop = stats.binomtest(len(df_B), len(df_estable), 0.5)
    print(f"  Proporción asignada a B = {len(df_B)/len(df_estable):.3f} (p-valor vs 0.5: {x_prop.pvalue:.3f})")

    conf_bonf = 1 - P.alpha / 3                      # nivel individual para confianza conjunta 1 − α
    gA = df_A.loc[df_A["cache_hit"] == 0, "gpu_ms"].dropna()
    gB = df_B.loc[df_B["cache_hit"] == 0, "gpu_ms"].dropna()
    A_i, A_l = df_A.dropna(subset=["intentos"]), df_A.dropna(subset=["latencia_ms"])   # filas válidas para (ii) y (iii)
    B_i, B_l = df_B.dropna(subset=["intentos"]), df_B.dropna(subset=["latencia_ms"])
    f3 = {
        "(i) GPU media (ms)": ic_dif_welch(gA, gB, conf_bonf),
        "(ii) P(intentos>1)": ic_dif_newcombe((A_i["intentos"] > 1).sum(), len(A_i),
                                             (B_i["intentos"] > 1).sum(), len(B_i), conf_bonf),
        "(iii) P(L>tau)": ic_dif_newcombe((A_l["latencia_ms"] > P.tau).sum(), len(A_l),
                                         (B_l["latencia_ms"] > P.tau).sum(), len(B_l), conf_bonf),
    }
    print(f"\nDiferencias B − A, IC Bonferroni (cada uno al {conf_bonf:.4f}):")
    for nombre, (d, lo, hi) in f3.items():
        veredicto = "B menor (mejor)" if hi < 0 else ("B mayor (peor)" if lo > 0 else "sin diferencia clara")
        print(f"  {nombre:20s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {veredicto}")
    # Regla de adopción: al menos una mejora con intervalo negativo y ningún intervalo que indique deterioro.
    params["F3"] = f3
    return {"f3": f3, "conf_bonf": conf_bonf, "gA": gA, "gB": gB}


# =============================================================================
# F4 · Cobertura de los IC de Wald y Wilson
# =============================================================================
def f4_cobertura(P, rng, params):
    """Cobertura efectiva de los IC de Wald y de Wilson con R4 conteos Binomial(n, p) por configuración."""
    filas = []
    for n in NS_F4:
        for p in PS_F4:
            conteos_binom = rng.binomial(n, p, R4)
            _, inferior_wald, superior_wald = ic_wald(conteos_binom, n, P.conf)
            _, inferior_wilson, superior_wilson = ic_wilson(conteos_binom, n, P.conf)
            # Fracción de intervalos que contienen la proporción verdadera.
            cubre_wald = (inferior_wald <= p) & (p <= superior_wald)
            cubre_wilson = (inferior_wilson <= p) & (p <= superior_wilson)
            filas.append({"n": n, "p": p, "Wald": cubre_wald.mean(), "Wilson": cubre_wilson.mean()})
    tabla_f4 = pd.DataFrame(filas)
    print(tabla_f4.pivot(index="p", columns="n", values=["Wald", "Wilson"]).round(3))
    params["metodo_ic_prop"] = "wilson"
    return {"tabla_f4": tabla_f4}


def figura_8(F4, P, ruta):
    """Figura 8: cobertura efectiva de Wald y Wilson frente a la proporción verdadera, para cada n."""
    tabla_f4 = F4["tabla_f4"]
    fig, axs = plt.subplots(1, len(NS_F4), figsize=(15, 3.5), sharey=True)
    for ax, n in zip(axs, NS_F4):
        sub = tabla_f4[tabla_f4["n"] == n]
        ax.plot(sub["p"], sub["Wald"], "o-", label="Wald")
        ax.plot(sub["p"], sub["Wilson"], "s-", label="Wilson")
        ax.axhline(P.conf, color="k", ls="--", label=f"nominal {P.conf:.0%}")
        ax.set_xscale("log")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("proporción verdadera p")
    axs[0].set_ylabel("cobertura efectiva")
    axs[0].legend(fontsize=8)
    fig.suptitle("Figura 8. Cobertura efectiva de los IC para una proporción")
    plt.figtext(0.01, -0.05, "Con p pequeña o n pequeño Wald cubre mucho menos de lo nominal; "
                "Wilson se mantiene cerca. Por eso usamos Wilson en E1, F3 y G2.", fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta / "figura8_wald_wilson.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# F5 · Tamaño de muestra
# =============================================================================
def f5_tamano_muestra(F3, P, rng):
    """Tamaño de muestra por versión para que el semiancho de F3(i) sea ≤ 1 ms: n ≥ z²(s²_A + s²_B)/1²,
    con z de Bonferroni, verificado remuestreando n observaciones por versión."""
    gA, gB, conf_bonf = F3["gA"], F3["gB"], F3["conf_bonf"]
    z_bonf = stats.norm.ppf(1 - (P.alpha / 3) / 2)
    sA2, sB2 = gA.var(ddof=1), gB.var(ddof=1)
    n_req = int(np.ceil(z_bonf**2 * (sA2 + sB2) / 1.0**2))   # se redondea hacia arriba
    print(f"sA² = {sA2:.1f}, sB² = {sB2:.1f}, z = {z_bonf:.4f}  ->  n por versión = {n_req}")

    semis = []
    for _ in range(R_F5):
        muestra_A = rng.choice(gA.to_numpy(), n_req)
        muestra_B = rng.choice(gB.to_numpy(), n_req)
        d, lo, hi = ic_dif_welch(muestra_A, muestra_B, conf_bonf)
        semis.append((hi - lo) / 2)
    semis = np.array(semis)
    print(f"Semiancho simulado: mediana={np.median(semis):.3f} ms, "
          f"P(semiancho <= 1) = {np.mean(semis <= 1):.3f}")
    return {"z_bonf": z_bonf, "n_req": n_req, "semis": semis}


# =============================================================================
# F6 · Validación con la segunda ventana
# =============================================================================
def f6_validacion(ruta_datos, df_estable, dur_estable, P):
    """Limpia solicitudes_validacion.csv con la función de A1 (único acceso a ese archivo) y compara cada
    parámetro con la ventana de referencia: Welch para medias, Newcombe para proporciones y normal para la tasa."""
    df_val = limpiar(pd.read_csv(ruta_datos / "solicitudes_validacion.csv"))
    tv = df_val.t_llegada_s.to_numpy()
    cuentas_val = np.histogram(tv, bins=np.arange(0, 1200 + 10, 10))[0]
    d_val = ((cuentas_val - cuentas_val.mean())**2 / cuentas_val.mean()).sum()
    p_disp_val = stats.chi2.sf(d_val, len(cuentas_val) - 1)
    print(f"Dispersión por 10 s p={p_disp_val:.4f}. Ventana de validación completa de 1200 s.")

    ve, vv = f6_datos(df_estable, dur_estable, P.tau), f6_datos(df_val, 1200, P.tau)
    f6_cambios = {}
    for metrica in ["gpu", "red", "lat"]:
        a_f6, b_f6 = ve[metrica], vv[metrica]
        f6_cambios[metrica] = ic_dif_welch(a_f6, b_f6, P.conf)
        print(metrica, "cambio (IC Welch)", f6_cambios[metrica], "p KS", stats.ks_2samp(a_f6, b_f6).pvalue)
    for metrica in ["reintentos", "cache", "violaciones"]:
        a_f6, b_f6 = ve[metrica], vv[metrica]
        f6_cambios[metrica] = ic_dif_newcombe(a_f6.sum(), len(a_f6), b_f6.sum(), len(b_f6), P.conf)
        print(metrica, "cambio (IC Newcombe)", f6_cambios[metrica])
    dif_tasa = vv["tasa"] - ve["tasa"]
    ee_tasa = np.sqrt(len(df_val) / 1200**2 + len(df_estable) / dur_estable**2)   # Var(n/T) = λ/T ≈ n/T²
    f6_cambios["tasa"] = (dif_tasa, dif_tasa - P.z_conf * ee_tasa, dif_tasa + P.z_conf * ee_tasa)
    # Carga ofrecida en GPUs equivalentes: λ·(1 − h)·E[G | no caché].
    carga_validacion = vv["tasa"] * (1 - vv["cache"].mean()) * vv["gpu"].mean() / 1000
    carga_referencia = ve["tasa"] * (1 - ve["cache"].mean()) * ve["gpu"].mean() / 1000
    print("Tasas de ambas ventanas (sol/s), diferencia e IC:", ve["tasa"], vv["tasa"], f6_cambios["tasa"])
    print(f"Carga ofrecida {carga_referencia:.4f} → {carga_validacion:.4f} GPUs equivalentes. F6 solo evalúa sensibilidad, G se calibra con producción.")
    return {"df_val": df_val, "p_disp_val": p_disp_val, "f6_cambios": f6_cambios,
            "carga_referencia": carga_referencia, "carga_validacion": carga_validacion}
