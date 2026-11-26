"""Parte F: inferencia, distribución muestral y comparación de versiones (Figuras 7 y 8)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .estadistica import ic_wald, ic_wilson, ic_media_t, ic_dif_welch, ic_dif_newcombe
from .parte_a import limpiar

M_F1 = 10_000                 # muestras por cada n en F1
N_GRAFICA_F1 = [5, 30, 200, 2000]
N_BUSQUEDA_F1 = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
K_LOTES = 30                  # número de lotes en F2
R4 = 10_000                   # réplicas por caso en F4
NS_F4 = [10, 30, 100, 500]
PS_F4 = [0.01, 0.02, 0.05, 0.10, 0.30, 0.50]
R_F5 = 2000                   # réplicas para verificar F5


def autocorr(x, lag):
    x = x - x.mean()
    return np.sum(x[:-lag] * x[lag:]) / np.sum(x * x)


def dif_estandarizada(a, b):
    """Diferencia de medias estandarizada; menos de 0.1 en valor absoluto se considera balanceado."""
    return (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)


def f6_datos(d, dur, tau):
    gpu = d.loc[d.cache_hit.eq(0), "gpu_ms"].dropna()
    red = d.red_ms.dropna()
    lat = d.latencia_ms.dropna()
    intentos = d.intentos.dropna()
    cache_d = d.cache_hit.dropna()
    return {"tasa": len(d) / dur, "gpu": gpu, "red": red, "lat": lat, "reintentos": intentos.gt(1),
            "cache": cache_d.eq(1), "violaciones": lat.gt(tau)}


def f1_distribucion_muestral(df_estable, P, rng):
    """La población es la latencia del régimen estable; se muestrea con reemplazo."""
    poblacion = df_estable["latencia_ms"].dropna().to_numpy()
    mu_pob, sd_pob = poblacion.mean(), poblacion.std()

    def medias_y_cobertura(n, M=M_F1, bloque=1000):
        """M medias de tamaño n y la proporción de IC normales que contienen la media real."""
        medias, cubre = [], 0
        for _ in range(M // bloque):             # por bloques para no llenar la memoria
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

    # se busca el menor n con cobertura >= conf - 0.01
    objetivo = P.conf - 0.01
    n_min = None
    coberturas = {}
    for n in N_BUSQUEDA_F1:
        _, cob = medias_y_cobertura(n)
        coberturas[n] = cob
        if cob >= objetivo:
            n_min = n
            break
    print(f"Población: media = {mu_pob:.2f} ms, desv = {sd_pob:.2f}, asimetría = {stats.skew(poblacion):.2f}, "
          f"curtosis = {stats.kurtosis(poblacion):.2f}")
    print(f"Menor n con cobertura >= {objetivo:.2f}: n = {n_min} (cobertura {coberturas[n_min]:.4f})"
          if n_min else f"Ningún n de la lista llega a {objetivo:.2f}")
    return {"mu_pob": mu_pob, "sd_pob": sd_pob, "tabla_f1": tabla_f1, "medias_por_n": medias_por_n,
            "n_min": n_min, "coberturas": coberturas}


def figura_7(F1, ruta):
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


def f2_ic_latencia(df_estable, P):
    """IC t clásico contra IC por medias de lotes (los lotes se arman dentro de cada tramo)."""
    lat_t = df_estable.sort_values("t_llegada_s")["latencia_ms"].dropna().to_numpy()
    m_cl, lo_cl, hi_cl = ic_media_t(lat_t, P.conf)

    acf = {k: autocorr(lat_t, k) for k in [1, 2, 5, 10, 50]}
    banda = P.z_conf / np.sqrt(lat_t.size)      # límite aproximado si no hubiera autocorrelación

    lotes = []
    for _, tramo in df_estable.sort_values("t_llegada_s").groupby("segmento"):
        lat_tramo = tramo["latencia_ms"].dropna().to_numpy()
        k_tramo = max(2, round(K_LOTES * len(lat_tramo) / len(lat_t)))   # lotes en proporción al tramo
        lotes += np.array_split(lat_tramo, k_tramo)
    medias_lote = np.array([b.mean() for b in lotes])
    m_bm, lo_bm, hi_bm = ic_media_t(medias_lote, P.conf)

    print("Autocorrelación de la latencia:", {k: round(v, 3) for k, v in acf.items()}, f"(límite ±{banda:.3f})")
    print(f"Autocorrelación lag 1 entre medias de lote: {autocorr(medias_lote, 1):.3f}")
    print(f"IC clásico {P.conf:.0%}:   {m_cl:.2f} [{lo_cl:.2f}, {hi_cl:.2f}]")
    print(f"IC por lotes {P.conf:.0%}: {m_bm:.2f} [{lo_bm:.2f}, {hi_bm:.2f}]  ({(hi_bm-lo_bm)/(hi_cl-lo_cl):.2f} veces más ancho)")
    return {"m_cl": m_cl, "lo_cl": lo_cl, "hi_cl": hi_cl, "acf": acf, "medias_lote": medias_lote,
            "m_bm": m_bm, "lo_bm": lo_bm, "hi_bm": hi_bm}


def f3_comparar_versiones(df_estable, P, params):
    """Tres diferencias B - A con Bonferroni (cada IC al nivel 1 - alpha/3)."""
    df_A = df_estable[df_estable["version"] == "A"]
    df_B = df_estable[df_estable["version"] == "B"]

    smd = {col: dif_estandarizada(df_A[col], df_B[col]) for col in ["tam_kb", "red_ms", "cache_hit"]}
    x_prop = stats.binomtest(len(df_B), len(df_estable), 0.5)

    conf_bonf = 1 - P.alpha / 3
    gA = df_A.loc[df_A["cache_hit"] == 0, "gpu_ms"].dropna()
    gB = df_B.loc[df_B["cache_hit"] == 0, "gpu_ms"].dropna()
    A_i, A_l = df_A.dropna(subset=["intentos"]), df_A.dropna(subset=["latencia_ms"])
    B_i, B_l = df_B.dropna(subset=["intentos"]), df_B.dropna(subset=["latencia_ms"])
    f3 = {
        "(i) GPU media (ms)": ic_dif_welch(gA, gB, conf_bonf),
        "(ii) P(intentos>1)": ic_dif_newcombe((A_i["intentos"] > 1).sum(), len(A_i),
                                             (B_i["intentos"] > 1).sum(), len(B_i), conf_bonf),
        "(iii) P(L>tau)": ic_dif_newcombe((A_l["latencia_ms"] > P.tau).sum(), len(A_l),
                                         (B_l["latencia_ms"] > P.tau).sum(), len(B_l), conf_bonf),
    }
    filas = []
    for nombre, (d, lo, hi) in f3.items():
        veredicto = "B menor (mejor)" if hi < 0 else ("B mayor (peor)" if lo > 0 else "sin diferencia clara")
        filas.append({"comparación": nombre, "B - A": d, "IC inf": lo, "IC sup": hi, "conclusión": veredicto})
    tabla_F3 = pd.DataFrame(filas).set_index("comparación")

    print("Balance (diferencia estandarizada): " + ", ".join(f"{k} {v:+.3f}" for k, v in smd.items())
          + f" | proporción en B = {len(df_B)/len(df_estable):.3f} (p = {x_prop.pvalue:.3f} contra 0.5)")
    print(f"Cada IC al {conf_bonf:.4f} para tener {P.conf:.0%} de confianza conjunta")
    params["F3"] = f3
    return {"f3": f3, "tabla_F3": tabla_F3, "smd": smd, "conf_bonf": conf_bonf, "gA": gA, "gB": gB}


def f4_cobertura(P, rng, params):
    filas = []
    for n in NS_F4:
        for p in PS_F4:
            conteos_binom = rng.binomial(n, p, R4)
            _, inferior_wald, superior_wald = ic_wald(conteos_binom, n, P.conf)
            _, inferior_wilson, superior_wilson = ic_wilson(conteos_binom, n, P.conf)
            cubre_wald = (inferior_wald <= p) & (p <= superior_wald)
            cubre_wilson = (inferior_wilson <= p) & (p <= superior_wilson)
            filas.append({"n": n, "p": p, "Wald": cubre_wald.mean(), "Wilson": cubre_wilson.mean()})
    tabla_f4 = pd.DataFrame(filas)
    params["metodo_ic_prop"] = "wilson"
    return {"tabla_f4": tabla_f4}


def figura_8(F4, P, ruta):
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
                "Wilson se mantiene cerca. Por eso se usa Wilson para las proporciones del proyecto.", fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta / "figura8_wald_wilson.png", dpi=150, bbox_inches="tight")
    plt.show()


def f5_tamano_muestra(F3, P, rng):
    """n por versión para que el semiancho de F3(i) sea <= 1 ms: n >= z^2 (sA^2 + sB^2)."""
    gA, gB, conf_bonf = F3["gA"], F3["gB"], F3["conf_bonf"]
    z_bonf = stats.norm.ppf(1 - (P.alpha / 3) / 2)
    sA2, sB2 = gA.var(ddof=1), gB.var(ddof=1)
    n_req = int(np.ceil(z_bonf**2 * (sA2 + sB2) / 1.0**2))

    # se comprueba simulando con ese n
    semis = []
    for _ in range(R_F5):
        muestra_A = rng.choice(gA.to_numpy(), n_req)
        muestra_B = rng.choice(gB.to_numpy(), n_req)
        d, lo, hi = ic_dif_welch(muestra_A, muestra_B, conf_bonf)
        semis.append((hi - lo) / 2)
    semis = np.array(semis)
    print(f"sA² = {sA2:.1f}, sB² = {sB2:.1f}, z = {z_bonf:.4f}, n por versión = {n_req}")
    print(f"Con ese n el semiancho simulado tiene mediana {np.median(semis):.3f} ms y P(semiancho <= 1) = {np.mean(semis <= 1):.3f}")
    return {"z_bonf": z_bonf, "n_req": n_req, "semis": semis}


def f6_validacion(ruta_datos, df_estable, dur_estable, P):
    """Limpia solicitudes_validacion.csv igual que en A1 y compara cada parámetro con la ventana original."""
    df_val = limpiar(pd.read_csv(ruta_datos / "solicitudes_validacion.csv"))
    tv = df_val.t_llegada_s.to_numpy()
    cuentas_val = np.histogram(tv, bins=np.arange(0, 1200 + 10, 10))[0]
    d_val = ((cuentas_val - cuentas_val.mean())**2 / cuentas_val.mean()).sum()
    p_disp_val = stats.chi2.sf(d_val, len(cuentas_val) - 1)

    ve, vv = f6_datos(df_estable, dur_estable, P.tau), f6_datos(df_val, 1200, P.tau)
    f6_cambios = {}
    p_ks = {}
    for metrica in ["gpu", "red", "lat"]:                       # medias: Welch
        a_f6, b_f6 = ve[metrica], vv[metrica]
        f6_cambios[metrica] = ic_dif_welch(a_f6, b_f6, P.conf)
        p_ks[metrica] = stats.ks_2samp(a_f6, b_f6).pvalue
    for metrica in ["reintentos", "cache", "violaciones"]:      # proporciones: Newcombe
        a_f6, b_f6 = ve[metrica], vv[metrica]
        f6_cambios[metrica] = ic_dif_newcombe(a_f6.sum(), len(a_f6), b_f6.sum(), len(b_f6), P.conf)
    dif_tasa = vv["tasa"] - ve["tasa"]
    ee_tasa = np.sqrt(len(df_val) / 1200**2 + len(df_estable) / dur_estable**2)   # Var(n/T) aprox. n/T^2
    f6_cambios["tasa"] = (dif_tasa, dif_tasa - P.z_conf * ee_tasa, dif_tasa + P.z_conf * ee_tasa)

    referencia = {"gpu": ve["gpu"].mean(), "red": ve["red"].mean(), "lat": ve["lat"].mean(),
                  "reintentos": ve["reintentos"].mean(), "cache": ve["cache"].mean(),
                  "violaciones": ve["violaciones"].mean(), "tasa": ve["tasa"]}
    tabla_F6 = pd.DataFrame([{"parámetro": k, "referencia": referencia[k], "validación - referencia": d,
                              "IC inf": lo, "IC sup": hi, "cambió": not (lo <= 0 <= hi), "p KS": p_ks.get(k, np.nan)}
                             for k, (d, lo, hi) in f6_cambios.items()]).set_index("parámetro")
    # carga en GPUs equivalentes: lambda (1 - h) E[G | no caché]
    carga_validacion = vv["tasa"] * (1 - vv["cache"].mean()) * vv["gpu"].mean() / 1000
    carga_referencia = ve["tasa"] * (1 - ve["cache"].mean()) * ve["gpu"].mean() / 1000
    print(f"Ventana de validación: {len(df_val)} solicitudes, dispersión por 10 s p = {p_disp_val:.4f}")
    print(f"Carga ofrecida: {carga_referencia:.4f} GPUs en la referencia y {carga_validacion:.4f} en la validación")
    return {"df_val": df_val, "p_disp_val": p_disp_val, "f6_cambios": f6_cambios, "tabla_F6": tabla_F6,
            "carga_referencia": carga_referencia, "carga_validacion": carga_validacion}
