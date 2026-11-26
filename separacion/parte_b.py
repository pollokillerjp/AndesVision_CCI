"""Parte B · Llegadas, fallos y reintentos."""
import numpy as np
import pandas as pd
from scipy import stats

B_KS_B1 = 599        # réplicas del KS calibrado por Montecarlo (B1)
B_SIM_B3 = 10_000    # muestras de la simulación calibrada de B3
N_MC_B4 = 200_000    # réplicas de Montecarlo de B4


# =============================================================================
# B1 · Proceso de llegadas
# =============================================================================
def b1_llegadas(df, df_estable, segmentos, P, rng):
    """Estima la tasa de llegadas del régimen estable y valida el modelo de Poisson por dos vías:
    tiempos entre llegadas frente a la exponencial y conteos por segundo frente a la Poisson."""
    # Las diferencias se calculan por tramo para no incluir el hueco que deja el pico entre los dos segmentos.
    dt_list = []
    for seg in ['estable_1', 'estable_2']:
        df_seg = df_estable[df_estable['segmento'] == seg].sort_values('t_llegada_s')
        dt = np.diff(df_seg['t_llegada_s'].values)
        dt_list.extend(dt[dt > 0])

    dt_arr = np.array(dt_list)
    n_dt = len(dt_arr)
    mean_dt = np.mean(dt_arr)
    lambda_hat = 1.0 / mean_dt                       # EMV de la tasa: inverso del tiempo medio entre llegadas

    # IC normal para la tasa, con Var(λ̂) ≈ λ̂²/n.
    se_lambda = lambda_hat / np.sqrt(n_dt)
    ic_lambda = (lambda_hat - P.z_conf * se_lambda, lambda_hat + P.z_conf * se_lambda)

    # Vía 1: Kolmogorov-Smirnov de los tiempos entre llegadas frente a Exponencial(λ̂).
    ks_stat, ks_pval = stats.kstest(dt_arr, 'expon', args=(0, mean_dt))

    # Vía 2: conteos en intervalos de 1 s frente a Poisson(λ̂ · 1 s).
    counts_1s = []
    for seg in ['estable_1', 'estable_2']:
        df_seg = df_estable[df_estable['segmento'] == seg]
        bins = np.arange(np.floor(df_seg['t_llegada_s'].min()), np.ceil(df_seg['t_llegada_s'].max()), 1.0)
        c, _ = np.histogram(df_seg['t_llegada_s'], bins=bins)
        counts_1s.extend(c)
    counts_1s = np.array(counts_1s)
    mean_c = np.mean(counts_1s)
    var_c = np.var(counts_1s, ddof=1)
    disp_index = var_c / mean_c                      # ≈ 1 si los conteos son Poisson

    # Chi-cuadrado de bondad de ajuste; la última clase acumula la cola y se usan solo clases con esperado ≥ 5.
    obs_counts, bin_edges = np.histogram(counts_1s, bins=np.arange(0, int(np.percentile(counts_1s, 99.5)) + 2))
    exp_probs = stats.poisson.pmf(bin_edges[:-1], mean_c)
    exp_probs[-1] += 1.0 - np.sum(exp_probs)
    exp_counts = exp_probs * len(counts_1s)
    valid_mask = exp_counts >= 5
    chi2_stat = np.sum((obs_counts[valid_mask] - exp_counts[valid_mask])**2 / exp_counts[valid_mask])
    dof = np.sum(valid_mask) - 1 - 1                 # clases − 1 − parámetro estimado (λ)
    chi2_pval = 1.0 - stats.chi2.cdf(chi2_stat, dof)

    print(f"Estimación puntual lambda = {lambda_hat:.4f} sol/s")
    print(f"IC al {int(P.conf*100)}%: [{ic_lambda[0]:.4f}, {ic_lambda[1]:.4f}] sol/s")
    print(f"Vía 1 - Test KS en delta_t: D = {ks_stat:.4f}, p-valor = {ks_pval:.4f} ")
    print(f"Vía 2 - Conteos en 1s: media = {mean_c:.3f}, var = {var_c:.3f}, índice de dispersión = {disp_index:.3f}")
    print(f"        Test Chi2: stat = {chi2_stat:.2f}, gl = {dof}, p-valor = {chi2_pval:.4f} ")

    # KS calibrado por Montecarlo: cada réplica vuelve a estimar la tasa, como se hizo con los datos (prueba de Lilliefors).
    d_boot = np.empty(B_KS_B1)
    for j in range(B_KS_B1):
        x_b1 = np.sort(rng.exponential(mean_dt, n_dt))
        F_b1 = stats.expon.cdf(x_b1, scale=x_b1.mean())
        d_boot[j] = max(np.max(np.arange(1, n_dt + 1) / n_dt - F_b1), np.max(F_b1 - np.arange(n_dt) / n_dt))
    p_ks_b1 = (1 + (d_boot >= ks_stat).sum()) / (B_KS_B1 + 1)
    # Dispersión bilateral de conteos por segundo con intervalos completos dentro de cada tramo estable.
    conteos_b1 = np.concatenate([np.histogram(tramo.t_llegada_s, bins=np.arange(np.ceil(a), np.floor(fin) + 1))[0]
                                 for a, fin, nom in segmentos if nom != "pico" for tramo in [df[df.segmento == nom]]])
    dis_b1 = (len(conteos_b1) - 1) * conteos_b1.var(ddof=1) / conteos_b1.mean()
    p_dis_b1 = min(1, 2 * min(stats.chi2.cdf(dis_b1, len(conteos_b1) - 1), stats.chi2.sf(dis_b1, len(conteos_b1) - 1)))
    print(f"KS calibrado (MC): p={p_ks_b1:.4f} | dispersión Poisson bilateral: p={p_dis_b1:.4f}")
    return {"lambda_hat": lambda_hat, "ic_lambda": ic_lambda, "se_lambda": se_lambda, "n_dt": n_dt, "mean_dt": mean_dt,
            "ks_stat": ks_stat, "ks_pval": ks_pval, "mean_c": mean_c, "var_c": var_c, "disp_index": disp_index,
            "chi2_stat": chi2_stat, "dof": dof, "chi2_pval": chi2_pval, "d_boot": d_boot,
            "p_ks_b1": p_ks_b1, "p_dis_b1": p_dis_b1}


# =============================================================================
# B2 · Bloques de 100 solicitudes
# =============================================================================
def b2_bloques(df, rng):
    """Cuenta solicitudes con reintento por bloque de 100 en cada régimen y contrasta la sobredispersión
    frente a Binomial(100, p̂). Los bloques se forman dentro de un mismo tramo para no mezclar condiciones."""
    bloques = {}
    filas_b2 = []
    for reg, d in df.groupby("regimen"):
        grupos = []
        for _, tramo in d.groupby("segmento", sort=False):
            valores = tramo.intentos.to_numpy()
            cantidad_completa = (len(valores) // 100) * 100
            bloques_de_100 = valores[:cantidad_completa].reshape(-1, 100)
            sin_faltantes = ~np.isnan(bloques_de_100).any(axis=1)        # se descartan bloques con intentos inválidos
            bloques_validos = bloques_de_100[sin_faltantes]
            grupos.extend((bloques_validos > 1).sum(axis=1))
        x_b = np.asarray(grupos)
        bloques[reg] = x_b
        pb = x_b.mean() / 100
        esperada = 100 * pb * (1 - pb)                                   # varianza binomial
        # Distribución de referencia: 4000 conjuntos de bloques Binomial(100, p̂).
        simulados = rng.binomial(100, pb, (4000, len(x_b)))
        pboot = simulados.mean(axis=1) / 100
        obs = x_b.var(ddof=1) / esperada                                 # razón varianza observada / binomial
        var_binomial_sim = 100 * pboot * (1 - pboot)
        dispersion = simulados.var(axis=1, ddof=1) / var_binomial_sim
        # Valor p bilateral: réplicas que se alejan de 1 al menos tanto como la observación.
        extremos = np.sum(abs(dispersion - 1) >= abs(obs - 1))
        p_b2 = (1 + extremos) / (len(dispersion) + 1)
        filas_b2.append((reg, len(x_b), pb, x_b.var(ddof=1), esperada, p_b2))
    tabla_b2 = pd.DataFrame(filas_b2, columns=["régimen", "bloques", "p", "var observada", "var binomial", "p MC"])
    return {"bloques": bloques, "tabla_b2": tabla_b2}


# =============================================================================
# B3 · Estimador de máxima verosimilitud de `intentos`
# =============================================================================
def b3_intentos(df_estable, P, rng):
    """EMV de la probabilidad de fallo por intento (variable censurada en 3) con IC de Fisher,
    y simulación calibrada que compara el EMV con el estimador de momentos."""
    k_vals = df_estable['intentos'].dropna().values
    n1 = np.sum(k_vals == 1)
    n2 = np.sum(k_vals == 2)
    n3 = np.sum(k_vals == 3)
    n_tot = n1 + n2 + n3

    # Llegar al tercer intento revela dos fallos previos; el resultado del tercer intento no se observa (censura).
    p_mle = (n2 + 2 * n3) / (n1 + 2 * n2 + 2 * n3)
    info_fisher = (n2 + 2 * n3) / (p_mle**2) + (n1 + n2) / ((1.0 - p_mle)**2)
    se_p = 1.0 / np.sqrt(info_fisher)
    ic_p = (p_mle - P.z_conf * se_p, p_mle + P.z_conf * se_p)

    # Simulación calibrada con p = p̂_MLE: B_SIM_B3 muestras del mismo tamaño que la observada.
    prob_geom = [1.0 - p_mle, p_mle * (1.0 - p_mle), p_mle**2]
    sim_k = rng.choice([1, 2, 3], size=(B_SIM_B3, n_tot), p=prob_geom)

    sn1 = np.sum(sim_k == 1, axis=1)
    sn2 = np.sum(sim_k == 2, axis=1)
    sn3 = np.sum(sim_k == 3, axis=1)
    sim_mle = (sn2 + 2 * sn3) / (sn1 + 2 * sn2 + 2 * sn3)

    # Estimador de momentos: resuelve E[K] = 1 + p + p² igualado al promedio de intentos.
    skbar = (sn1 + 2 * sn2 + 3 * sn3) / n_tot
    sim_mom = (-1.0 + np.sqrt(np.maximum(0.0, 1.0 - 4.0 * (1.0 - skbar)))) / 2.0

    print(f"Conteos observados: n1={n1}, n2={n2}, n3={n3} (Total={n_tot})")
    print(f"p_MLE = {p_mle:.5f} (SE = {se_p:.5f})")
    print(f"IC al {int(P.conf*100)}%: [{ic_p[0]:.5f}, {ic_p[1]:.5f}]")
    print(f"Simulación Calibrada ({B_SIM_B3} réplicas):")
    print(f"  MLE:      Sesgo = {np.mean(sim_mle) - p_mle:.2e}, MSE = {np.mean((sim_mle - p_mle)**2):.2e}")
    print(f"  Momentos: Sesgo = {np.mean(sim_mom) - p_mle:.2e}, MSE = {np.mean((sim_mom - p_mle)**2):.2e}")
    print(f"  Eficiencia Relativa Var(MLE)/Var(Mom) = {np.var(sim_mle, ddof=1)/np.var(sim_mom, ddof=1):.4f}")
    return {"n1": n1, "n2": n2, "n3": n3, "n_tot": n_tot, "p_mle": p_mle, "se_p": se_p, "ic_p": ic_p,
            "sim_mle": sim_mle, "sim_mom": sim_mom}


# =============================================================================
# B4 · Ventana de 10 s
# =============================================================================
def b4_ventana_10s(lambda_hat, p_mle, P, rng):
    """P(Z ≥ k) para el conteo de solicitudes con reintento en 10 s, por adelgazamiento de Poisson
    y verificación por Montecarlo, con k = ⌈μ̂⌉ + r."""
    mu_hat = 10.0 * lambda_hat * p_mle
    k_val = int(np.ceil(mu_hat) + P.r)

    # Probabilidad analítica: P(Z ≥ k) = 1 − F(k − 1).
    prob_analitica = 1.0 - stats.poisson.cdf(k_val - 1, mu_hat)

    # Montecarlo: llegadas Poisson(10λ̂) y, de ellas, las que requieren reintento ~ Binomial(llegadas, p̂).
    sol_10s = rng.poisson(10.0 * lambda_hat, size=N_MC_B4)
    fallos_10s = rng.binomial(sol_10s, p_mle)
    prob_mc = np.mean(fallos_10s >= k_val)

    print(f"mu_hat = {mu_hat:.4f}")
    print(f"Parámetro r = {P.r} -> Umbral k = {k_val}")
    print(f"Probabilidad Analítica P(Z >= {k_val}) = {prob_analitica:.5f}")
    print(f"Probabilidad Monte Carlo P(Z >= {k_val}) = {prob_mc:.5f}")
    print(f"Discrepancia absoluta = {abs(prob_analitica - prob_mc):.5f}")
    return {"mu_hat": mu_hat, "k_val": k_val, "prob_analitica": prob_analitica, "prob_mc": prob_mc}
