"""Parte A: auditoría de los datos, regímenes y estadística descriptiva (Figura 1)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

TOL = 0.021          # tolerancia (ms) porque los tiempos vienen redondeados a centésimas
T_VENTANA = 1200.0   # duración de la ventana (s)
MARGEN = 30          # segundos que se quitan junto a cada corte en la prueba de sensibilidad

POLITICAS = {
    "duplicada": "Quitar copia exacta",
    "id_repetido_no_exacto": "Conservar la medición físicamente válida",
    "tam_escala": "Dividir por 1024, patrón de centésimas reconstruido",
    "tam_vacio": "Excluir por variable",
    "red_faltante": "Excluir solo de análisis de red",
    "gpu_faltante": "Excluir solo de análisis de GPU",
    "red_negativa": "Corregir signo",
    "cache_con_gpu": "Invalidar caché y GPU",
    "latencia_imposible": "Invalidar latencia",
    "exito_sin_3_intentos": "Invalidar intentos y éxito",
}


# ---------------------------------------------------------------------------
# funciones de apoyo
# ---------------------------------------------------------------------------
def detectar_inconsistencias(d):
    """Marca en cada fila las reglas físicas que no se cumplen (una fila puede tener varias)."""
    # algunos tamaños están en bytes: al dividir por 1024 quedan en rango y con dos decimales
    tam_convertido_kb = d.tam_kb / 1024
    centesimas_kb = tam_convertido_kb * 100
    en_rango = tam_convertido_kb.between(1, 2000)
    en_centesimas = np.isclose(centesimas_kb, np.rint(centesimas_kb), atol=1e-6, rtol=0)
    escala = (d.tam_kb > 10000) & en_rango & en_centesimas

    duplicada = d.duplicated()
    ids_sin_copias = d.loc[~duplicada, "id_solicitud"]
    id_repetido = ids_sin_copias.duplicated(keep=False).reindex(d.index, fill_value=False)
    cache_con_gpu = (d.cache_hit == 1) & (d.gpu_ms > 0)
    # L = red + espera + G y la espera no puede ser negativa
    resto_ms = d.latencia_ms - d.red_ms.abs() - d.gpu_ms
    latencia_imposible = (d.latencia_ms < 0) | ((resto_ms < -TOL) & ~cache_con_gpu)
    return pd.DataFrame({
        "duplicada": duplicada,
        "id_repetido_no_exacto": ~duplicada & id_repetido,
        "tam_escala": escala, "tam_vacio": d.tam_kb.isna(),
        "red_faltante": d.red_ms.isna(), "gpu_faltante": d.gpu_ms.isna(),
        "red_negativa": d.red_ms < 0,
        "cache_con_gpu": cache_con_gpu,
        "latencia_imposible": latencia_imposible,
        "exito_sin_3_intentos": (d.exito == 0) & (d.intentos < 3),
    })


def limpiar(d, corregir_escala=True):
    """Aplica los tratamientos de A1. Los valores inválidos quedan como NaN pero la fila se conserva,
    porque la solicitud sí llegó. También se usa en F6."""
    b = detectar_inconsistencias(d)
    out = d.loc[~b.duplicada].copy()
    b = b.loc[out.index]
    out["red_ms"] = out["red_ms"].abs()
    out.loc[b.tam_escala, "tam_kb"] = out.loc[b.tam_escala, "tam_kb"] / 1024 if corregir_escala else np.nan
    out.loc[b.cache_con_gpu, ["cache_hit", "gpu_ms"]] = np.nan
    out.loc[b.latencia_imposible, "latencia_ms"] = np.nan
    out.loc[b.exito_sin_3_intentos, ["intentos", "exito"]] = np.nan
    # si un id aparece dos veces con datos distintos, se deja la fila con más datos válidos
    out = (out.assign(_validos=out.notna().sum(axis=1)).sort_values("_validos", ascending=False, kind="stable")
           .drop_duplicates("id_solicitud").drop(columns="_validos")
           .sort_values("t_llegada_s", kind="stable").reset_index(drop=True))
    return out


def cifras_clave(d, nombre):
    return pd.Series({
        "solicitudes": len(d),
        "tam_kb medio": d.tam_kb.mean(),
        "latencia media (ms)": d.latencia_ms.mean(),
        "gpu_ms medio sin caché": d.loc[d.cache_hit == 0, "gpu_ms"].mean(),
        "red_ms medio": d.red_ms.mean(),
        "P(intentos > 1)": (d.intentos.dropna() > 1).mean(),
        "P(caché)": d.cache_hit.mean(),
    }, name=nombre)


def prueba_dispersion(tiempos, inicio, fin, ancho):
    """Índice de dispersión de los conteos en bins de `ancho` segundos. Si las llegadas son Poisson,
    D sigue una chi-cuadrado con k-1 grados de libertad. Devuelve (k, D, p)."""
    bordes = np.arange(inicio, fin + 1e-9, ancho)
    conteos = np.histogram(tiempos, bins=bordes)[0]
    D = ((conteos - conteos.mean()) ** 2).sum() / conteos.mean()
    return len(conteos), D, stats.chi2.sf(D, len(conteos) - 1)


def ll_poisson(n, duracion):
    """Log-verosimilitud de un tramo Poisson con n llegadas, evaluada en lambda = n/duracion (sin constantes)."""
    n = np.asarray(n, float); duracion = np.asarray(duracion, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, n * np.log(n / duracion) - n, 0.0)


def buscar_dos_cambios(tiempos, resolucion, T=T_VENTANA):
    """Prueba todos los pares de cortes en una rejilla y devuelve el de mayor verosimilitud: (logL, t1, t2)."""
    n_total = len(tiempos)
    rejilla = np.arange(resolucion, T, resolucion)
    N_antes = np.searchsorted(tiempos, rejilla)          # llegadas antes de cada punto de la rejilla
    mejor = (-np.inf, None, None)
    for i in range(len(rejilla) - 1):
        a, Na = rejilla[i], N_antes[i]
        b, Nb = rejilla[i + 1:], N_antes[i + 1:]           # todos los posibles segundos cortes de una vez
        ll = ll_poisson(Na, a) + ll_poisson(Nb - Na, b - a) + ll_poisson(n_total - Nb, T - b)
        j = ll.argmax()
        if ll[j] > mejor[0]:
            mejor = (float(ll[j]), float(a), float(b[j]))
    return mejor


def mejor_corte(tiempos, rejilla, inicio, fin, margen=5.0):
    """Mejor corte extra dentro de [inicio, fin) y cuánto sube la log-verosimilitud con él."""
    candidatos = rejilla[(rejilla > inicio + margen) & (rejilla < fin - margen)]
    N0, N1 = np.searchsorted(tiempos, [inicio, fin])
    Nc = np.searchsorted(tiempos, candidatos)
    ll = ll_poisson(Nc - N0, candidatos - inicio) + ll_poisson(N1 - Nc, fin - candidatos)
    j = ll.argmax()
    return float(candidatos[j]), float(ll[j] - ll_poisson(N1 - N0, fin - inicio))


def comparar(a, b, dur_a, dur_b):
    """Compara dos tramos variable por variable y devuelve las medias y el valor p de cada prueba."""
    filas = []
    # dado el total, las llegadas del tramo a son binomiales con p = dur_a / (dur_a + dur_b)
    p_tasa = stats.binomtest(len(a), len(a) + len(b), dur_a / (dur_a + dur_b)).pvalue
    filas.append(["Tasa de llegadas (sol/s)", len(a) / dur_a, len(b) / dur_b, "Binomial exacta condicional", p_tasa])
    ga = a.loc[a.cache_hit == 0, "gpu_ms"].dropna()
    gb = b.loc[b.cache_hit == 0, "gpu_ms"].dropna()
    filas.append(["gpu_ms medio sin caché", ga.mean(), gb.mean(), "Kolmogorov-Smirnov", stats.ks_2samp(ga, gb).pvalue])
    for col, etiqueta in [("intentos", "P(intentos > 1)"), ("cache_hit", "P(caché)")]:
        xa, xb = a[col].dropna(), b[col].dropna()
        ia, ib = (xa > 1, xb > 1) if col == "intentos" else (xa == 1, xb == 1)
        tabla = [[ia.sum(), (~ia).sum()], [ib.sum(), (~ib).sum()]]
        filas.append([etiqueta, ia.mean(), ib.mean(), "Chi-cuadrado 2x2", stats.chi2_contingency(tabla)[1]])
    for col, etiqueta in [("red_ms", "red_ms medio"), ("tam_kb", "tam_kb medio"), ("latencia_ms", "latencia media (ms)")]:
        xa, xb = a[col].dropna(), b[col].dropna()
        filas.append([etiqueta, xa.mean(), xb.mean(), "Kolmogorov-Smirnov", stats.ks_2samp(xa, xb).pvalue])
    return pd.DataFrame(filas, columns=["Variable", "Tramo 1", "Tramo 2", "Prueba", "valor p"])


def cifras_estable(d, dur):
    return pd.Series({"tasa (sol/s)": len(d) / dur,
                      "gpu_ms medio sin caché": d.loc[d.cache_hit == 0, "gpu_ms"].mean(),
                      "P(intentos > 1)": (d.intentos.dropna() > 1).mean(),
                      "latencia media (ms)": d.latencia_ms.mean()})


def describir(x):
    x = x.dropna()
    return pd.Series({"n": len(x), "media": x.mean(), "mediana": x.median(), "desv. est.": x.std(ddof=1),
                      "RIC": x.quantile(.75) - x.quantile(.25),
                      "asimetría": stats.skew(x), "curtosis (exceso)": stats.kurtosis(x)})


# ---------------------------------------------------------------------------
# A1: auditoría y limpieza
# ---------------------------------------------------------------------------
def leer_solicitudes(ruta_datos):
    crudo = pd.read_csv(ruta_datos / "solicitudes.csv")
    print(f"solicitudes.csv: {len(crudo)} filas")
    return crudo


def a1_auditoria(crudo):
    banderas = detectar_inconsistencias(crudo)
    sin_dup = crudo[~banderas.duplicada]
    saltos_atras = int((sin_dup.t_llegada_s.diff() < 0).sum())   # llegadas fuera de orden

    # evidencia para justificar cada tratamiento
    tam_alt = crudo.loc[banderas.tam_escala, "tam_kb"] / 1024
    lat_mala = crudo.loc[banderas.latencia_imposible, "latencia_ms"]
    con_fallo = crudo[crudo.exito == 0]
    evidencia = {
        "tam_max_sin_corregir": crudo.loc[~banderas.tam_escala, "tam_kb"].max(),
        "tam_corregido_min": tam_alt.min(), "tam_corregido_max": tam_alt.max(),
        "error_centesimas": np.max(np.abs(tam_alt * 100 - np.rint(tam_alt * 100))),
        "latencias_id_repetido": crudo.loc[banderas.id_repetido_no_exacto, "latencia_ms"].tolist(),
        "latencias_menos_1": int((lat_mala == -1).sum()),
        "fallos_con_3_intentos": bool((con_fallo.intentos == 3).all()),
    }
    print(f"Tamaños en bytes: {int(banderas.tam_escala.sum())} filas. Al dividir por 1024 quedan entre "
          f"{evidencia['tam_corregido_min']:.2f} y {evidencia['tam_corregido_max']:.2f} KB con dos decimales "
          f"(el máximo de los demás es {evidencia['tam_max_sin_corregir']:.2f} KB)")
    print(f"Latencias imposibles: {len(lat_mala)}, de ellas {evidencia['latencias_menos_1']} con valor -1")
    print(f"Id repetido con datos distintos, latencias: {evidencia['latencias_id_repetido']}")
    print(f"Toda solicitud con exito = 0 tiene 3 intentos: {evidencia['fallos_con_3_intentos']}")

    tabla_A1 = pd.DataFrame({"filas": banderas.sum(), "tratamiento": pd.Series(POLITICAS)})
    tabla_A1.loc["saltos hacia atrás"] = [saltos_atras, "Ordenar por tiempo"]
    return {"banderas": banderas, "saltos_atras": saltos_atras, "evidencia": evidencia, "tabla_A1": tabla_A1}


def a1_limpiar(crudo):
    df = limpiar(crudo)
    # sensibilidad: correlación tamaño-GPU corrigiendo los tamaños o dejándolos fuera
    comparacion_tam = {}
    for etiqueta, datos in {"corregido": df, "tamaños dudosos omitidos": limpiar(crudo, False)}.items():
        muestra = datos.loc[datos.cache_hit.eq(0), ["tam_kb", "gpu_ms"]].dropna()
        comparacion_tam[etiqueta] = muestra.corr().loc["tam_kb", "gpu_ms"]
    print(f"Solicitudes únicas después de limpiar: {len(df)}")
    print(f"Correlación tamaño-GPU: {comparacion_tam['corregido']:.4f} corrigiendo los tamaños, "
          f"{comparacion_tam['tamaños dudosos omitidos']:.4f} omitiéndolos")
    antes_despues = pd.concat([cifras_clave(crudo, "crudo"), cifras_clave(df, "limpio")], axis=1)
    return {"df": df, "comparacion_tam": comparacion_tam, "antes_despues": antes_despues}


# ---------------------------------------------------------------------------
# A2: regímenes
# ---------------------------------------------------------------------------
def a2_detectar_cambios(df):
    t_llegadas = df.t_llegada_s.to_numpy()
    n_total = len(t_llegadas)

    # ¿toda la ventana es un solo Poisson? índice de dispersión con varios anchos de bin
    filas = []
    for ancho in [5, 10, 20, 30, 60]:
        k, D, p = prueba_dispersion(t_llegadas, 0, T_VENTANA, ancho)
        filas.append({"ancho del bin (s)": ancho, "bins": k, "D": round(D, 1), "gl": k - 1, "valor p": p})
    tabla_dispersion = pd.DataFrame(filas)

    # modelos con 0, 1 y 2 cambios de tasa
    REJILLA = np.arange(0.5, T_VENTANA, 0.5)
    N_ANTES = np.searchsorted(t_llegadas, REJILLA)
    ll_0 = float(ll_poisson(n_total, T_VENTANA))
    ll_1_vec = ll_poisson(N_ANTES, REJILLA) + ll_poisson(n_total - N_ANTES, T_VENTANA - REJILLA)
    ll_1, cambio_unico = float(ll_1_vec.max()), float(REJILLA[ll_1_vec.argmax()])
    ll_2, TAU1, TAU2 = buscar_dos_cambios(t_llegadas, 0.5)

    # 3 cambios = el mejor corte extra dentro de alguno de los tramos
    SEGMENTOS = [(0.0, TAU1, "estable_1"), (TAU1, TAU2, "pico"), (TAU2, T_VENTANA, "estable_2")]
    ganancia_3 = max(mejor_corte(t_llegadas, REJILLA, a, b)[1] for a, b, _ in SEGMENTOS)
    modelos = pd.DataFrame({"cambios": [0, 1, 2, 3], "logL": [ll_0, ll_1, ll_2, ll_2 + ganancia_3]})
    modelos["parámetros"] = 2 * modelos.cambios + 1      # k+1 tasas y k instantes
    modelos["BIC"] = -2 * modelos.logL + modelos["parámetros"] * np.log(n_total)
    modelos["AIC"] = -2 * modelos.logL + 2 * modelos["parámetros"]

    # dentro de cada tramo: ¿hay dispersión extra o un cambio más que valga la pena?
    filas = []
    for a, b, nombre in SEGMENTOS:
        m = (t_llegadas >= a) & (t_llegadas < b)
        fila = {"tramo": nombre, "desde (s)": a, "hasta (s)": b, "llegadas": int(m.sum()), "tasa (sol/s)": round(m.sum() / (b - a), 3)}
        for ancho in [5, 10, 30, 60]:
            fila[f"p dispersión {ancho}s"] = round(prueba_dispersion(t_llegadas[m], a, b, ancho)[2], 3)
        corte, gan = mejor_corte(t_llegadas, REJILLA, a, b)
        fila["2·ganancia de un cambio extra"] = round(2 * gan, 2)
        fila["penalización BIC"] = round(2 * np.log(n_total), 2)   # un cambio extra agrega 2 parámetros
        filas.append(fila)
    tabla_tramos = pd.DataFrame(filas)

    # sensibilidad de los cortes a la rejilla
    sens = []
    for res in [0.25, 0.5, 1.0, 2.0, 5.0]:
        _, a, b = buscar_dos_cambios(t_llegadas, res)
        sens.append({"resolución (s)": res, "t1 (s)": a, "t2 (s)": b})
    tabla_resolucion = pd.DataFrame(sens)

    # instantes con logL a menos de 2 unidades del máximo, moviendo un corte y dejando fijo el otro
    N2 = np.searchsorted(t_llegadas, TAU2); N1 = np.searchsorted(t_llegadas, TAU1)
    perfil1 = ll_poisson(N_ANTES, REJILLA) + ll_poisson(N2 - N_ANTES, TAU2 - REJILLA) + ll_poisson(n_total - N2, T_VENTANA - TAU2)
    perfil2 = ll_poisson(N1, TAU1) + ll_poisson(N_ANTES - N1, REJILLA - TAU1) + ll_poisson(n_total - N_ANTES, T_VENTANA - REJILLA)
    ok1 = REJILLA[(REJILLA < TAU2) & (perfil1 >= ll_2 - 2)]
    ok2 = REJILLA[(REJILLA > TAU1) & (perfil2 >= ll_2 - 2)]

    p_disp = ", ".join(f"{r['ancho del bin (s)']} s: {r['valor p']:.2g}" for _, r in tabla_dispersion.iterrows())
    print(f"Índice de dispersión en toda la ventana (valor p por ancho de bin): {p_disp}")
    print(f"Cortes estimados: t1 = {TAU1} s y t2 = {TAU2} s (con un solo cambio sería t = {cambio_unico} s)")
    print(f"BIC elige {int(modelos.loc[modelos.BIC.idxmin(), 'cambios'])} cambios y AIC elige "
          f"{int(modelos.loc[modelos.AIC.idxmin(), 'cambios'])}")
    print(f"Con rejillas de 0.25 a 5 s: t1 entre {tabla_resolucion['t1 (s)'].min()} y {tabla_resolucion['t1 (s)'].max()} s, "
          f"t2 entre {tabla_resolucion['t2 (s)'].min()} y {tabla_resolucion['t2 (s)'].max()} s")
    print(f"Rango plausible (logL a menos de 2 del máximo): t1 en [{ok1.min()}, {ok1.max()}] s, t2 en [{ok2.min()}, {ok2.max()}] s")
    return {"t_llegadas": t_llegadas, "n_total": n_total, "tabla_dispersion": tabla_dispersion,
            "REJILLA": REJILLA, "N_ANTES": N_ANTES, "ll_0": ll_0, "ll_1": ll_1, "cambio_unico": cambio_unico,
            "ll_2": ll_2, "TAU1": TAU1, "TAU2": TAU2, "SEGMENTOS": SEGMENTOS, "modelos": modelos,
            "tabla_tramos": tabla_tramos, "tabla_resolucion": tabla_resolucion,
            "t1_plausible": (ok1.min(), ok1.max()), "t2_plausible": (ok2.min(), ok2.max())}


def a2_definir_regimenes(df, cambios):
    """Agrega 'segmento' y 'regimen' a df y compara los tramos estables entre sí y contra el pico."""
    TAU1, TAU2 = cambios["TAU1"], cambios["TAU2"]
    df["segmento"] = np.select([df.t_llegada_s < TAU1, df.t_llegada_s < TAU2], ["estable_1", "pico"], "estable_2")
    df["regimen"] = np.where(df.segmento == "pico", "pico", "estable")
    df_estable = df[df.regimen == "estable"].copy()     # lo que se usa de B en adelante
    df_reg2 = df[df.regimen == "pico"].copy()

    seg_estable_1, seg_pico, seg_estable_2 = (df[df.segmento == s] for s in ["estable_1", "pico", "estable_2"])
    ia1 = np.diff(seg_estable_1.t_llegada_s.to_numpy()); ia3 = np.diff(seg_estable_2.t_llegada_s.to_numpy())
    p_ks_tramos = stats.ks_2samp(ia1, ia3).pvalue
    comp_estables = comparar(seg_estable_1, seg_estable_2, TAU1, T_VENTANA - TAU2)
    comp_estable_pico = comparar(df_estable, seg_pico, TAU1 + (T_VENTANA - TAU2), TAU2 - TAU1)
    comparacion = pd.DataFrame({
        "estable_1": comp_estables["Tramo 1"], "estable_2": comp_estables["Tramo 2"], "p (1 vs 2)": comp_estables["valor p"],
        "estable": comp_estable_pico["Tramo 1"], "pico": comp_estable_pico["Tramo 2"], "p (estable vs pico)": comp_estable_pico["valor p"],
        "prueba": comp_estables["Prueba"]}).set_axis(comp_estables["Variable"])

    dur_estable = TAU1 + (T_VENTANA - TAU2)
    recortado = df_estable[(df_estable.t_llegada_s < TAU1 - MARGEN) | (df_estable.t_llegada_s >= TAU2 + MARGEN)]
    tabla_margen = pd.concat([cifras_estable(df_estable, dur_estable).rename("régimen estable"),
                              cifras_estable(recortado, dur_estable - 2 * MARGEN).rename(f"sin {MARGEN} s junto a los cambios")], axis=1)

    print(f"Régimen estable: {len(df_estable)} solicitudes en {dur_estable:.1f} s | pico: {len(df_reg2)} solicitudes")
    print(f"Tiempos entre llegadas de estable_1 vs estable_2 (KS): p = {p_ks_tramos:.3f}")
    print(f"Quitando {MARGEN} s junto a cada corte, la tasa pasa de {tabla_margen.iloc[0, 0]:.3f} a {tabla_margen.iloc[0, 1]:.3f} sol/s "
          f"y la latencia media de {tabla_margen.iloc[3, 0]:.2f} a {tabla_margen.iloc[3, 1]:.2f} ms")
    return {"df_estable": df_estable, "df_reg2": df_reg2, "dur_estable": dur_estable, "p_ks_tramos": p_ks_tramos,
            "comp_estables": comp_estables, "comp_estable_pico": comp_estable_pico, "comparacion": comparacion,
            "tabla_margen": tabla_margen}


def exportar_limpio(df, ruta):
    df.to_csv(ruta / "solicitudes_limpio.csv", index=False)


# ---------------------------------------------------------------------------
# A3: descriptivos y Figura 1
# ---------------------------------------------------------------------------
def a3_descriptivos(df):
    """Descriptivos por variable, régimen y versión. Cada uno usa solo los valores válidos de su columna."""
    variables = {
        "latencia_ms": df.latencia_ms,
        "gpu_ms (todas)": df.gpu_ms,
        "gpu_ms (sin caché)": df.gpu_ms.where(df.cache_hit == 0),
        "red_ms": df.red_ms,
        "tam_kb": df.tam_kb,
    }
    filas = []
    for nombre, serie in variables.items():
        for reg in ["estable", "pico"]:
            for ver in ["A", "B", "A+B"]:
                m = (df.regimen == reg) & ((df.version == ver) if ver != "A+B" else True)
                filas.append(describir(serie[m]).rename((nombre, reg, ver)))
    tabla_A3 = pd.DataFrame(filas)
    tabla_A3.index = pd.MultiIndex.from_tuples(tabla_A3.index, names=["variable", "régimen", "versión"])
    tabla_A3["n"] = tabla_A3["n"].astype(int)
    # versión corta para mostrar: solo A y B y las medidas que pide el enunciado
    resumen = tabla_A3[tabla_A3.index.get_level_values("versión") != "A+B"][["media", "mediana", "desv. est.", "asimetría", "curtosis (exceso)"]]
    return {"tabla_A3": tabla_A3, "tabla_A3_resumen": resumen}


def figura_1(df, cambios, ruta):
    TAU1, TAU2, SEGMENTOS, t_llegadas = cambios["TAU1"], cambios["TAU2"], cambios["SEGMENTOS"], cambios["t_llegadas"]
    ancho = 10
    bordes = np.arange(0, T_VENTANA + 1e-9, ancho)
    centros = bordes[:-1] + ancho / 2
    g10 = df.groupby(pd.cut(df.t_llegada_s, bordes, right=False), observed=False)
    bordes30 = np.arange(0, T_VENTANA + 1e-9, 30)
    centros30 = bordes30[:-1] + 15
    g30 = df.groupby(pd.cut(df.t_llegada_s, bordes30, right=False), observed=False)
    g30_nc = df[df.cache_hit == 0].groupby(pd.cut(df.loc[df.cache_hit == 0, "t_llegada_s"], bordes30, right=False), observed=False)

    fig, ejes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax in ejes:
        ax.axvspan(TAU1, TAU2, color="tab:red", alpha=0.08, label="Régimen pico")
        for c in (TAU1, TAU2):
            ax.axvline(c, color="tab:red", ls="--", lw=1)
        ax.grid(alpha=0.3)

    ejes[0].step(centros, g10.size().to_numpy() / ancho, where="mid", color="tab:blue", label="Observado (bins de 10 s)")
    for a, b, _ in SEGMENTOS:
        m = (t_llegadas >= a) & (t_llegadas < b)
        ejes[0].hlines(m.sum() / (b - a), a, b, color="black", lw=2)
    ejes[0].plot([], [], color="black", lw=2, label="Tasa ajustada por tramo (EMV)")
    ejes[0].set_ylabel("Llegadas (sol/s)")
    ejes[0].set_title("(a) Tasa de llegadas")

    ejes[1].plot(centros, g10.latencia_ms.median().to_numpy(), color="tab:green", label="Mediana")
    ejes[1].plot(centros, g10.latencia_ms.quantile(0.95).to_numpy(), color="tab:orange", label="Percentil 95")
    ejes[1].set_ylabel("Latencia (ms)")
    ejes[1].set_title("(b) Latencia extremo a extremo (bins de 10 s)")

    ejes[2].plot(centros30, 100 * g30.intentos.apply(lambda s: (s.dropna() > 1).mean()).to_numpy(), marker="o", color="tab:purple", label="% con más de 1 intento")
    ejes[2].set_ylabel("Solicitudes (%)")
    ejes[2].set_title("(c) Proporción de solicitudes con reintentos (bins de 30 s)")

    ejes[3].plot(centros30, g30_nc.gpu_ms.mean().to_numpy(), marker="o", color="tab:brown", label="Media de gpu_ms sin caché")
    ejes[3].set_ylabel("Tiempo de GPU (ms)")
    ejes[3].set_title("(d) Tiempo de cómputo en GPU de las solicitudes no cacheadas (bins de 30 s)")
    ejes[3].set_xlabel("Tiempo desde el inicio de la ventana (s)")

    for ax in ejes:
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Figura 1. Evolución temporal del sistema en la ventana de 20 minutos", fontsize=13, y=0.995)
    fig.text(0.01, 0.002,
             "El área sombreada identifica el pico estimado. Los paneles comparan tráfico, latencia, reintentos y cómputo.\n"
             "Fuera de esa zona el sistema se comporta igual al inicio y al final: ese es el régimen estable que se usa en las partes B a F.",
             fontsize=9, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(ruta / "figura1_evolucion.png", dpi=150)
    plt.show()
