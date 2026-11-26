"""Parte E · Monitoreo y decisión de alertas (Figura 6)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .estadistica import ic_wilson

REGLAS = ["D1", "D2", "D1_y_D2", "D1_o_D2"]
N_SIM_BETA = 20_000      # sorteos de sensibilidad y FPR por regla (E2 y E4)
MINUTOS_DIA = 1440


def vpp(sens, fpr, prev):
    """Valor predictivo positivo por el teorema de Bayes: P(incidente | alerta) con prevalencia `prev`."""
    alerta_verdadera = sens * prev
    alerta_falsa = fpr * (1 - prev)
    prob_alerta = alerta_verdadera + alerta_falsa
    return alerta_verdadera / prob_alerta


def leer_monitoreo(ruta_datos, P):
    """Lee monitoreo.csv y agrega las reglas combinadas «D1 y D2» y «D1 o D2»."""
    mon = pd.read_csv(ruta_datos / "monitoreo.csv")
    mon["D1_y_D2"] = (mon["D1"] & mon["D2"]).astype(int)
    mon["D1_o_D2"] = (mon["D1"] | mon["D2"]).astype(int)
    inc = mon["incidente"] == 1
    print(f"Minutos: {len(mon)}, con incidente: {inc.sum()} ({inc.mean():.3%})  "
          f"<- prevalencia del entorno de PRUEBAS, no de producción (π = {P.pi_prod})")
    return mon


# =============================================================================
# E1 · Sensibilidad y tasa de falsos positivos
# =============================================================================
def e1_sensibilidad(mon, P):
    """Sensibilidad (denominador: minutos con incidente) y FPR (denominador: minutos sin incidente)
    de cada regla, con IC de Wilson."""
    inc = mon["incidente"] == 1
    filas = []
    for regla in REGLAS:
        x_s, n_s = mon.loc[inc, regla].sum(), inc.sum()
        x_f, n_f = mon.loc[~inc, regla].sum(), (~inc).sum()
        s, sl, su = ic_wilson(x_s, n_s, P.conf)
        f, fl, fu = ic_wilson(x_f, n_f, P.conf)
        filas.append({"regla": regla, "x_sens": x_s, "n_sens": n_s, "x_fp": x_f, "n_fp": n_f,
                      "sens": s, "sens_lo": sl, "sens_hi": su, "fpr": f, "fpr_lo": fl, "fpr_hi": fu})
    e1 = pd.DataFrame(filas).set_index("regla")
    return {"e1": e1}


# =============================================================================
# E2 · Valor predictivo positivo en producción
# =============================================================================
def e2_vpp(mon, E1, P, rng):
    """VPP en producción con IC por propagación Montecarlo (sensibilidad y FPR sorteadas de sus Beta de Jeffreys),
    frente al VPP que se obtendría con la prevalencia del archivo de pruebas."""
    e1 = E1["e1"]
    prev_pruebas = (mon["incidente"] == 1).mean()
    filas = []
    for regla, f in e1.iterrows():
        s_sim = rng.beta(f.x_sens + 0.5, f.n_sens - f.x_sens + 0.5, N_SIM_BETA)
        f_sim = rng.beta(f.x_fp + 0.5, f.n_fp - f.x_fp + 0.5, N_SIM_BETA)
        v_sim = vpp(s_sim, f_sim, P.pi_prod)
        lo, hi = np.quantile(v_sim, [P.alpha / 2, 1 - P.alpha / 2])
        filas.append({"regla": regla, "VPP_produccion": vpp(f.sens, f.fpr, P.pi_prod),
                      "lo": lo, "hi": hi, "VPP_en_monitoreo.csv": vpp(f.sens, f.fpr, prev_pruebas)})
    e2 = pd.DataFrame(filas).set_index("regla")
    return {"e2": e2, "prev_pruebas": prev_pruebas}


# =============================================================================
# E3 · Independencia condicional de D1 y D2
# =============================================================================
def e3_independencia(mon, E1, E2, P):
    """Prueba exacta de Fisher de D1 × D2 dentro de cada estado y efecto de suponer independencia sobre el VPP."""
    e1, e2 = E1["e1"], E2["e2"]
    inc = mon["incidente"] == 1
    resultados = {}
    for nombre, mask in [("con incidente", inc), ("sin incidente", ~inc)]:
        sub = mon[mask]
        tabla = pd.crosstab(sub["D1"], sub["D2"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        odds, p_f = stats.fisher_exact(tabla.values)
        p11 = (sub["D1"] & sub["D2"]).mean()
        p_producto = sub["D1"].mean() * sub["D2"].mean()     # valor esperado de p11 si fueran independientes
        print(f"[{nombre}] tabla D1×D2:\n{tabla}\n  P(ambas)={p11:.4f}  producto={p_producto:.4f}  "
              f"odds ratio={odds:.2f}  Fisher p={p_f:.4g}")
        resultados[nombre] = {"p11": p11, "producto": p_producto, "odds_ratio": odds, "p_fisher": p_f}

    # VPP de «D1 y D2» si se supusiera independencia: sensibilidades y FPR se multiplican.
    s_ind = e1.loc["D1", "sens"] * e1.loc["D2", "sens"]
    f_ind = e1.loc["D1", "fpr"] * e1.loc["D2", "fpr"]
    vpp_ind = vpp(s_ind, f_ind, P.pi_prod)
    print(f"\nVPP de 'D1 y D2' en producción: real={e2.loc['D1_y_D2','VPP_produccion']:.4f}  "
          f"suponiendo independencia={vpp_ind:.4f}")
    return {"por_estado": resultados, "vpp_independencia": vpp_ind}


# =============================================================================
# E4 · Falsas alarmas por día
# =============================================================================
def e4_falsas_alarmas(E1, P, rng, params):
    """P(X ≤ kmax) con X ~ Binomial(n0, FPR) falsas alarmas diarias, propagando la incertidumbre de la FPR.
    Una regla cumple si el percentil α de esa probabilidad sigue siendo ≥ 0.90; se recomienda la de mayor
    sensibilidad entre las que cumplen."""
    e1 = E1["e1"]
    n0 = int(round(MINUTOS_DIA * (1 - P.pi_prod)))    # minutos sin incidente en un día de producción
    filas = []
    dist_prob = {}
    for regla, f in e1.iterrows():
        # Con pocos falsos positivos la FPR es incierta: cada FPR sorteada da una probabilidad diaria distinta.
        f_sim = rng.beta(f.x_fp + 0.5, f.n_fp - f.x_fp + 0.5, N_SIM_BETA)
        prob_sim = stats.binom.cdf(P.kmax, n0, f_sim)
        dist_prob[regla] = prob_sim
        filas.append({"regla": regla, "FA_esperadas_dia": n0 * f.fpr,
                      "P(X<=kmax) puntual": stats.binom.cdf(P.kmax, n0, f.fpr),
                      "P(X<=kmax) peor caso": np.quantile(prob_sim, P.alpha),
                      "P(cumple)": np.mean(prob_sim >= 0.90),
                      "sens": f.sens})
    e4 = pd.DataFrame(filas).set_index("regla")
    e4["cumple"] = e4["P(X<=kmax) peor caso"] >= 0.90
    print(f"kmax = {P.kmax}, minutos sin incidente por día = {n0}")
    print(e4.round(4).to_string())

    candidatas = e4[e4["cumple"]]
    if len(candidatas):
        recomendada = candidatas["sens"].idxmax()
        print(f"Regla recomendada (cumple y detecta más incidentes): {recomendada}")
    else:
        # Sin reglas admisibles se reporta la de mayor probabilidad puntual.
        recomendada = e4["P(X<=kmax) puntual"].idxmax()
        print(f"NINGUNA regla cumple la exigencia. La más cercana es {recomendada} "
              f"(P = {e4.loc[recomendada, 'P(X<=kmax) puntual']:.3f}); hay que reportarlo así.")
    params["regla_alerta"] = recomendada
    return {"e4": e4, "dist_prob": dist_prob, "recomendada": recomendada, "n0": n0}


def figura_6(E4, P, ruta):
    """Figura 6: incertidumbre de P(X ≤ kmax) para cada regla frente a la exigencia de 0.90.
    El modelo binomial supone independencia entre minutos; una correlación temporal positiva aumentaría la varianza."""
    dist_prob = E4["dist_prob"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([dist_prob[r] for r in REGLAS], showfliers=False)
    ax.set_xticks(range(1, len(REGLAS) + 1), REGLAS)
    ax.axhline(0.90, color="red", ls="--", label="exigencia: 0.90")
    ax.set_ylabel(f"P(falsas alarmas en un día ≤ {P.kmax})")
    ax.set_xlabel("regla de alerta")
    ax.set_title("Figura 6. Probabilidad de no superar kmax falsas alarmas diarias")
    ax.legend()
    plt.figtext(0.01, -0.04, "Cada caja refleja la incertidumbre de la FPR estimada; las reglas cuya "
                "caja queda sobre la línea roja cumplen la exigencia del equipo de guardia.", fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta / "figura6_alertas.png", dpi=150, bbox_inches="tight")
    plt.show()
