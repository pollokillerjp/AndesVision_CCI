"""Parte D: dependencias, esperanza condicional y regresión de G sobre el tamaño (Figuras 4 y 5)."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .estadistica import ic_wilson, ic_media_t, txt_ic, a_nativo

CLASES = ["pequeña", "mediana", "grande"]


def nadaraya_watson(x, y, rejilla, ancho):
    """Estimación no paramétrica de E[y | x] con núcleo gaussiano."""
    w = np.exp(-0.5 * ((rejilla[:, None] - x[None, :]) / ancho) ** 2)
    return (w * y).sum(axis=1) / w.sum(axis=1)


def descomponer(df_, grupos):
    """Comprueba las leyes de esperanza y varianza total de gpu_ms por grupos.
    Se usa ddof=0 para que la suma dé exacta."""
    por_grupo = df_.groupby(grupos, observed=True)
    pesos = por_grupo.size() / len(df_)
    medias_ms = por_grupo.gpu_ms.mean()
    varianzas = por_grupo.gpu_ms.var(ddof=0)
    media_total_ms = df_.gpu_ms.mean()
    varianza_total = df_.gpu_ms.var(ddof=0)
    media_ponderada_ms = (pesos * medias_ms).sum()
    dentro = (pesos * varianzas).sum()                          # E[Var(G | grupo)]
    entre = (pesos * (medias_ms - media_total_ms) ** 2).sum()   # Var(E[G | grupo])
    return {
        "E[G]": media_total_ms,
        "Σ P·E[G|grupo]": media_ponderada_ms,
        "Var(G)": varianza_total,
        "E[Var(G|grupo)]": dentro,
        "Var(E[G|grupo])": entre,
        "suma": dentro + entre,
        "fracción explicada": entre / varianza_total,
    }


def breusch_pagan(residuos, X_):
    """Breusch-Pagan: se regresa e^2 sobre X y LM = n R^2 se compara con una chi-cuadrado."""
    e2 = residuos**2
    b_aux = np.linalg.solve(X_.T @ X_, X_.T @ e2)
    r_aux = e2 - X_ @ b_aux
    R2_aux = 1 - (r_aux @ r_aux) / ((e2 - e2.mean()) @ (e2 - e2.mean()))
    LM = len(e2) * R2_aux
    return LM, stats.chi2.sf(LM, X_.shape[1] - 1)


def d1_tamano_cache(df_estable, P, params):
    d1 = df_estable.dropna(subset=["tam_kb", "cache_hit"]).copy()
    # right=False: si el tamaño es igual a un corte, queda en la clase de arriba
    d1["clase"] = pd.cut(d1.tam_kb, [-np.inf, P.t1, P.t2, np.inf], right=False, labels=CLASES)
    d1["cache"] = d1.cache_hit.astype(int)
    conteos = pd.crosstab(d1.clase, d1.cache, margins=True, margins_name="Total")
    conjunta = pd.crosstab(d1.clase, d1.cache, normalize="all", margins=True, margins_name="Total")

    filas = []
    for c in CLASES:
        sub = d1[d1.clase == c]
        p, lo, hi = ic_wilson(sub.cache.sum(), len(sub), P.conf)
        filas.append({"clase": c, "n": len(sub), "P(caché | clase)": p, "IC inf": lo, "IC sup": hi})
    cond_cache = pd.DataFrame(filas)
    cond_por_cache = pd.crosstab(d1.clase, d1.cache, normalize="columns")   # P(clase | caché)

    tab = pd.crosstab(d1.clase, d1.cache).to_numpy()
    chi2, p_ind_tam, gl, esperados = stats.chi2_contingency(tab)
    V = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    cache_peq, cache_gra = d1[d1.clase == "pequeña"].cache, d1[d1.clase == "grande"].cache
    dif = cache_peq.mean() - cache_gra.mean()
    ee = np.sqrt(cache_peq.mean()*(1-cache_peq.mean())/len(cache_peq) + cache_gra.mean()*(1-cache_gra.mean())/len(cache_gra))

    print(f"Clases: pequeña < {P.t1} KB <= mediana < {P.t2} KB <= grande")
    print(f"Independencia: chi-cuadrado = {chi2:.1f} con {gl} gl, p = {p_ind_tam:.3g}, V de Cramér = {V:.3f}")
    print(f"P(caché | pequeña) - P(caché | grande) = {txt_ic(dif, dif - P.z_conf*ee, dif + P.z_conf*ee, 4)} (Wald {P.conf:.0%})")
    params["p_cache_clase"] = {c: float(v) for c, v in zip(cond_cache.clase, cond_cache["P(caché | clase)"])}
    return {"d1": d1, "conteos": conteos, "conjunta": conjunta, "cond_cache": cond_cache, "cond_por_cache": cond_por_cache,
            "tab": tab, "chi2": chi2, "p_ind_tam": p_ind_tam, "V": V}


def figura_4(D1, P, ruta):
    d1, tab, cond_cache = D1["d1"], D1["tab"], D1["cond_cache"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    conj_mat = pd.crosstab(d1.clase, d1.cache, normalize="all").to_numpy()
    im = ax[0].imshow(conj_mat, cmap="Blues")
    for i in range(3):
        for j in range(2):
            ax[0].text(j, i, f"{conj_mat[i, j]:.3f}\n(n={tab[i, j]})", ha="center", va="center",
                       color="white" if conj_mat[i, j] > conj_mat.max() * 0.6 else "black", fontsize=10)
    ax[0].set_xticks([0, 1], ["sin caché (0)", "con caché (1)"])
    ax[0].set_yticks(range(3), [f"pequeña (<{P.t1} KB)", "mediana", f"grande (≥{P.t2} KB)"])
    ax[0].set(title="(a) P(clase, caché) en el régimen estable", xlabel="cache_hit", ylabel="Clase de tamaño")
    fig.colorbar(im, ax=ax[0], label="Probabilidad conjunta")
    ax[1].bar(CLASES, cond_cache["P(caché | clase)"], color="tab:blue", alpha=0.7, label="P(caché | clase)")
    ax[1].errorbar(CLASES, cond_cache["P(caché | clase)"],
                   yerr=[cond_cache["P(caché | clase)"] - cond_cache["IC inf"], cond_cache["IC sup"] - cond_cache["P(caché | clase)"]],
                   fmt="none", color="black", capsize=6, label=f"IC {P.conf:.0%} (Wilson)")
    ax[1].axhline(d1.cache.mean(), color="tab:red", ls="--", label="P(caché) global (si fueran independientes)")
    ax[1].set(title="(b) Probabilidad de caché según el tamaño", xlabel="Clase de tamaño", ylabel="P(caché | clase)", ylim=(0, 0.75))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Figura 4. Tamaño y caché NO son independientes: las imágenes pequeñas salen más de la caché", y=1.02)
    fig.text(0.01, -0.05, "Si fueran independientes, las tres barras quedarían sobre la línea roja. Sus intervalos no la tocan: "
             "la probabilidad de caché baja a medida que crece la imagen.", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta / "figura4_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


def d2_esperanza_condicional(df_estable, P):
    d2 = df_estable.dropna(subset=["tam_kb", "cache_hit", "gpu_ms"]).copy()
    d2["clase"] = pd.cut(d2.tam_kb, [-np.inf, P.t1, P.t2, np.inf], right=False, labels=CLASES)
    filas = []
    for c in CLASES:
        m, lo, hi = ic_media_t(d2.loc[(d2.clase == c) & (d2.cache_hit == 0), "gpu_ms"], P.conf)
        filas.append({"clase": c, "E[G | clase, no caché] (ms)": m, "IC inf": lo, "IC sup": hi})
    tabla_medias = pd.DataFrame(filas)

    x_all, g_all = d2.tam_kb.to_numpy(), d2.gpu_ms.to_numpy()
    ancho_nw = 1.06 * min(x_all.std(), stats.iqr(x_all) / 1.34) * len(x_all) ** (-1 / 5)   # regla de Silverman
    rejilla_nw = np.linspace(np.quantile(x_all, 0.01), np.quantile(x_all, 0.99), 150)
    EG_tam = nadaraya_watson(x_all, g_all, rejilla_nw, ancho_nw)          # con los ceros de caché
    nc2 = d2.cache_hit == 0
    EG_tam_nc = nadaraya_watson(x_all[nc2], g_all[nc2], rejilla_nw, ancho_nw)
    print(f"E[G | tam_kb] por Nadaraya-Watson con ancho de banda {ancho_nw:.1f} KB (curva en la Figura 5)")

    tabla_D2 = pd.DataFrame({
        "G (todas) | clase": descomponer(d2, "clase"),
        "G (todas) | clase y caché": descomponer(d2, ["clase", "cache_hit"]),
        "G (sin caché) | clase": descomponer(d2[nc2], "clase"),
    }).T
    return {"d2": d2, "tabla_medias": tabla_medias, "ancho_nw": ancho_nw, "rejilla_nw": rejilla_nw,
            "EG_tam": EG_tam, "EG_tam_nc": EG_tam_nc, "tabla_D2": tabla_D2}


def d3_mco(validas, P):
    """Mínimos cuadrados resolviendo (X'X) beta = X'y."""
    d3 = validas[(validas.cache_hit == 0) & validas.tam_kb.notna()]
    x, y = d3.tam_kb.to_numpy(), d3.gpu_ms.to_numpy()
    n_reg = len(y)
    X = np.column_stack([np.ones(n_reg), x])        # columna de unos para el intercepto
    XtX, Xty = X.T @ X, X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    e = y - X @ beta
    sigma2 = (e @ e) / (n_reg - 2)
    cov_ols = sigma2 * np.linalg.inv(XtX)
    ee_ols = np.sqrt(np.diag(cov_ols))
    tq = stats.t.ppf(1 - P.alpha / 2, n_reg - 2)
    R2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    print(f"b0 = {txt_ic(beta[0], beta[0]-tq*ee_ols[0], beta[0]+tq*ee_ols[0], 3)} ms")
    print(f"b1 = {txt_ic(beta[1], beta[1]-tq*ee_ols[1], beta[1]+tq*ee_ols[1], 5)} ms/KB  (IC t {P.conf:.0%})")
    print(f"sigma = {np.sqrt(sigma2):.3f} ms, R² = {R2:.4f}")
    print(f"Para la deducción a mano: n = {n_reg}, Σx = {x.sum():.2f}, Σx² = {(x**2).sum():.2f}, "
          f"Σy = {y.sum():.2f}, Σxy = {(x*y).sum():.2f}")
    return {"x": x, "y": y, "n_reg": n_reg, "X": X, "XtX": XtX, "beta": beta, "e": e, "sigma2": sigma2,
            "ee_ols": ee_ols, "tq": tq, "R2": R2}


def d3_residuos(mco):
    X, y, beta, e = mco["X"], mco["y"], mco["beta"], mco["e"]
    y_hat = X @ beta
    LM, p_bp = breusch_pagan(e, X)
    # pendiente de log Var contra log media por deciles; si da cerca de 2 el CV es constante (como una Gamma)
    dec = pd.qcut(y_hat, 10, labels=False)
    fv = pd.DataFrame({"mu": y_hat, "y": y}).groupby(dec).agg(media=("y", "mean"), var=("y", "var"))
    pendiente_var = np.polyfit(np.log(fv.media), np.log(fv["var"]), 1)[0]
    print(f"Breusch-Pagan: LM = {LM:.1f}, p = {p_bp:.3g} | asimetría de los residuos = {stats.skew(e):.3f}")
    print(f"Pendiente de log Var(G|x) contra log E[G|x] = {pendiente_var:.2f}")
    return {"y_hat": y_hat, "LM": LM, "p_bp": p_bp, "pendiente_var": pendiente_var}


def d3_correccion(mco, c2_cond, P, params):
    """Mínimos cuadrados ponderados con w = 1/mu^2 (iterando) y errores robustos HC3."""
    X, y, beta, e, XtX = mco["X"], mco["y"], mco["beta"], mco["e"], mco["XtX"]
    n_reg, ee_ols, tq = mco["n_reg"], mco["ee_ols"], mco["tq"]
    beta_w = beta.copy()
    for it in range(100):
        mu = X @ beta_w
        w = 1 / mu**2                                # donde se espera más varianza se da menos peso
        XtWX, XtWy = X.T @ (w[:, None] * X), X.T @ (w * y)
        nuevo = np.linalg.solve(XtWX, XtWy)
        if np.max(np.abs(nuevo - beta_w)) < 1e-12:
            beta_w = nuevo
            break
        beta_w = nuevo
    mu = X @ beta_w
    w = 1 / mu**2
    phi = (w * (y - mu) ** 2).sum() / (n_reg - 2)    # estimación de CV^2
    cov_wls = phi * np.linalg.inv(X.T @ (w[:, None] * X))
    ee_wls = np.sqrt(np.diag(cov_wls))
    r_std = np.sqrt(w) * (y - mu)
    LM_w, p_bp_w = breusch_pagan(r_std, X)

    # otra opción: dejar MCO y corregir solo los errores estándar (HC3)
    inv = np.linalg.inv(XtX)
    hii = np.einsum("ij,jk,ik->i", X, inv, X)
    cov_hc3 = inv @ (X.T @ ((e**2 / (1 - hii) ** 2)[:, None] * X)) @ inv
    ee_hc3 = np.sqrt(np.diag(cov_hc3))

    b0_g, b1_g, ee_cond = c2_cond["b0_g"], c2_cond["b1_g"], c2_cond["ee_cond"]
    tabla_D3 = pd.DataFrame({
        "β0": [beta[0], beta[0], beta_w[0], b0_g],
        "ee β0": [ee_ols[0], ee_hc3[0], ee_wls[0], ee_cond[0]],
        "β1": [beta[1], beta[1], beta_w[1], b1_g],
        "ee β1": [ee_ols[1], ee_hc3[1], ee_wls[1], ee_cond[1]],
    }, index=["MCO (supone varianza constante)", "MCO + errores robustos HC3", "MC ponderados, w = 1/μ² (corrección)", "EMV Gamma condicional (C2)"])
    tabla_D3[f"IC β1 {P.conf:.0%}"] = [f"[{b - tq*s:.5f}, {b + tq*s:.5f}]" for b, s in zip(tabla_D3["β1"], tabla_D3["ee β1"])]
    print(f"Ponderados: {it + 1} iteraciones, CV = {np.sqrt(phi):.3f}, Breusch-Pagan de los residuos ponderados p = {p_bp_w:.3f}")
    params["beta"] = {"b0": [beta_w[0], beta_w[0] - tq*ee_wls[0], beta_w[0] + tq*ee_wls[0]],
                      "b1": [beta_w[1], beta_w[1] - tq*ee_wls[1], beta_w[1] + tq*ee_wls[1]],
                      "cv": float(np.sqrt(phi)), "metodo": "MC ponderados iterados, w = 1/mu^2"}
    return {"beta_w": beta_w, "mu": mu, "w": w, "phi": phi, "ee_wls": ee_wls, "r_std": r_std,
            "LM_w": LM_w, "p_bp_w": p_bp_w, "ee_hc3": ee_hc3, "tabla_D3": tabla_D3}


def figura_5(mco, resid, corr, D2, rng, ruta):
    x, y, beta, e, n_reg = mco["x"], mco["y"], mco["beta"], mco["e"], mco["n_reg"]
    y_hat, beta_w, mu, r_std = resid["y_hat"], corr["beta_w"], corr["mu"], corr["r_std"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    idx = rng.choice(n_reg, size=min(3000, n_reg), replace=False)        # solo 3000 puntos para que se vea el gráfico
    ax[0].scatter(x[idx], y[idx], s=4, alpha=0.25, color="gray", label="Solicitudes no cacheadas (muestra de 3000)")
    xs = np.linspace(x.min(), np.quantile(x, 0.995), 200)
    ax[0].plot(xs, beta[0] + beta[1] * xs, "tab:blue", lw=2, label=f"MCO: {beta[0]:.1f} + {beta[1]:.4f}·tam")
    ax[0].plot(xs, beta_w[0] + beta_w[1] * xs, "tab:red", lw=2, ls="--", label=f"MC ponderados: {beta_w[0]:.1f} + {beta_w[1]:.4f}·tam")
    ax[0].plot(D2["rejilla_nw"], D2["EG_tam_nc"], "black", lw=1.5, ls=":", label="E[G | tam] no paramétrica (D2)")
    ax[0].set(title="(a) gpu_ms frente a tam_kb", xlabel="tam_kb (KB)", ylabel="gpu_ms (ms)", xlim=(0, np.quantile(x, 0.995)), ylim=(0, 200))
    ax[1].scatter(y_hat[idx], e[idx], s=4, alpha=0.25, color="tab:blue")
    cort = pd.qcut(y_hat, 15, labels=False)
    banda = pd.DataFrame({"f": y_hat, "e": e}).groupby(cort).agg(f=("f", "mean"), sd=("e", "std"))
    ax[1].plot(banda.f, 2 * banda.sd, "k-", label="±2 desv. est. por tramo")
    ax[1].plot(banda.f, -2 * banda.sd, "k-")
    ax[1].axhline(0, color="red", lw=1)
    ax[1].set(title="(b) Residuos MCO: el abanico se abre", xlabel="Valor ajustado (ms)", ylabel="Residuo (ms)")
    ax[2].scatter(mu[idx], r_std[idx], s=4, alpha=0.25, color="tab:red")
    banda_w = pd.DataFrame({"f": mu, "r": r_std}).groupby(pd.qcut(mu, 15, labels=False)).agg(f=("f", "mean"), sd=("r", "std"))
    ax[2].plot(banda_w.f, 2 * banda_w.sd, "k-", label="±2 desv. est. por tramo")
    ax[2].plot(banda_w.f, -2 * banda_w.sd, "k-")
    ax[2].axhline(0, color="red", lw=1)
    ax[2].set(title="(c) Residuos ponderados: ancho constante", xlabel="Valor ajustado (ms)", ylabel="Residuo ponderado √w·e")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Figura 5. El tiempo de GPU crece linealmente con el tamaño, pero su dispersión crece con la media", y=1.02)
    fig.text(0.01, -0.06, "La recta describe bien la media (coincide con la curva no paramétrica), pero los residuos de MCO se abren en abanico. "
             "Con pesos w = 1/μ² el abanico desaparece (panel c),\ncomo predice un tiempo de cómputo con coeficiente de variación constante.", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta / "figura5_regresion.png", dpi=150, bbox_inches="tight")
    plt.show()


def d4_correlacion(mco, D2, P):
    x, y, n_reg, beta, R2 = mco["x"], mco["y"], mco["n_reg"], mco["beta"], mco["R2"]
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    r_xy = np.corrcoef(x, y)[0, 1]
    zf = np.arctanh(r_xy)
    ee_z = 1 / np.sqrt(n_reg - 3)  # transformación z de Fisher
    r_lo, r_hi = np.tanh(zf - P.z_conf * ee_z), np.tanh(zf + P.z_conf * ee_z)
    rho_s = stats.spearmanr(x, y)[0]
    eta2_nc = D2["tabla_D2"].loc["G (sin caché) | clase", "fracción explicada"]
    print(f"Cov(tam, gpu) = {cov_xy:.2f} KB·ms, Pearson r = {txt_ic(r_xy, r_lo, r_hi, 4)} (IC {P.conf:.0%}), Spearman = {rho_s:.4f}")
    print(f"Cov/Var(tam) = {cov_xy/np.var(x, ddof=1):.5f} (igual a b1 = {beta[1]:.5f}) y r² = {r_xy**2:.4f} (igual a R² = {R2:.4f})")
    print(f"La clase explica {eta2_nc:.4f} de la varianza de G sin caché; el tamaño continuo explica r² = {r_xy**2:.4f}")
    return {"cov_xy": cov_xy, "r_xy": r_xy, "r_lo": r_lo, "r_hi": r_hi, "rho_s": rho_s}


def exportar_parametros(P, params, C1, C2, C1M, mco, corr, ruta, borrador=False):
    """Guarda los parámetros ajustados y los números de las deducciones de C1 y D3 en parametros_C_D.json."""
    x, y = mco["x"], mco["y"]
    salida = {"S": P.S, "borrador": borrador, "conf": P.conf, "t1": P.t1, "t2": P.t2, "N": P.N, "params": a_nativo(params),
              "deducciones": a_nativo({"h": C1["h_hat"], "k": C2["k_mv"], "theta": C2["th_mv"], "k_mm": C2["k_mm"], "theta_mm": C2["th_mm"],
                                       "E_G_formula": C1M["E_mv"], "Var_G_formula": C1M["V_mv"],
                                       "E_G_muestra": C1["G_todas"].mean(), "Var_G_muestra": C1["G_todas"].var(ddof=1),
                                       "n": mco["n_reg"], "sum_x": x.sum(), "sum_x2": (x**2).sum(), "sum_y": y.sum(), "sum_xy": (x*y).sum(),
                                       "xbar": x.mean(), "ybar": y.mean(), "Sxx": ((x-x.mean())**2).sum(), "Sxy": ((x-x.mean())*(y-y.mean())).sum(),
                                       "b0": mco["beta"][0], "b1": mco["beta"][1], "ee_b0": mco["ee_ols"][0], "ee_b1": mco["ee_ols"][1],
                                       "sigma": np.sqrt(mco["sigma2"]), "tq": mco["tq"],
                                       "b0_w": corr["beta_w"][0], "b1_w": corr["beta_w"][1], "cv": np.sqrt(corr["phi"])})}
    (ruta / "parametros_C_D.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf8")
    return salida
