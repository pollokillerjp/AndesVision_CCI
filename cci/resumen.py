"""Junta las cifras principales de todas las partes en resumen_consolidado.json."""
import json

import numpy as np
from scipy import stats

from .estadistica import a_nativo


def construir_resumen(R, P, ruta):
    A1, A1L, A2, REG = R["A1"], R["A1L"], R["A2"], R["REG"]
    df, df_estable, df_reg2 = A1L["df"], REG["df_estable"], REG["df_reg2"]
    mco, corr = R["D3_mco"], R["D3_corr"]
    E4 = R["E4"]
    n_reg = mco["n_reg"]
    tq_hc3 = stats.t.ppf(1 - P.alpha / 2, n_reg - 2)
    resumen_final = dict(
        datos=[len(R["crudo"]), len(R["mon"]), len(R["F6"]["df_val"])],
        a=dict(limpios=len(df_estable) + len(df_reg2), estable=len(df_estable), cortes=[A2["TAU1"], A2["TAU2"]],
               dur=REG["dur_estable"], tasas=[len(df[df.segmento == s]) / (fin - a) for a, fin, s in A2["SEGMENTOS"]],
               bic=A2["modelos"].BIC.tolist(), auditoria=A1["tabla_A1"]["filas"].to_dict(), pearson_sens=A1L["comparacion_tam"]),
        b=dict(tasa=R["B1"]["lambda_hat"], ic=R["B1"]["ic_lambda"], pks=R["B1"]["p_ks_b1"], pdisp=R["B1"]["p_dis_b1"],
               bloques=R["B2"]["tabla_b2"].to_dict("records"), p=R["B3"]["p_mle"], ic_p=R["B3"]["ic_p"],
               k=R["B4"]["k_val"], prob=R["B4"]["prob_analitica"], mc=R["B4"]["prob_mc"]),
        c=dict(h=R["C1"]["h_hat"], gpu=R["C1"]["G_c"].mean(), gamma=[R["C2"]["k_mv"], R["C2"]["th_mv"]],
               media_mixta=R["C1"]["G_todas"].mean(), red=R["C3_colas"]["p999_emp"], ic_red=R["C3_colas"]["ic_red"],
               c4=R["C4"]["res_N"], ks=R["C2_ajuste"]["p_ks_gam"]),
        d=dict(pearson=R["D4"]["r_xy"], ic=[R["D4"]["r_lo"], R["D4"]["r_hi"]], pendiente=mco["beta"][1],
               pendiente_hc3=[mco["beta"][1] - tq_hc3 * corr["ee_hc3"][1], mco["beta"][1] + tq_hc3 * corr["ee_hc3"][1]],
               explicada=R["D2"]["tabla_D2"].iloc[0]["fracción explicada"], v=R["D1"]["V"], p=R["D1"]["p_ind_tam"]),
        e=dict(regla=E4["recomendada"], prob=E4["e4"].loc[E4["recomendada"], "P(X<=kmax) puntual"],
               peor=E4["e4"].loc[E4["recomendada"], "P(X<=kmax) peor caso"],
               sensibilidad=R["E1"]["e1"].loc[E4["recomendada"], "sens"], cumple=bool(E4["e4"].cumple.any()), max=P.kmax),
        f=dict(diferencias=R["F3"]["f3"], latencia=[R["F2"]["m_bm"], R["F2"]["lo_bm"], R["F2"]["hi_bm"]],
               n_f5=R["F5"]["n_req"], n_tlc=R["F1"]["n_min"], validacion=R["F6"]["f6_cambios"],
               disp_validacion=R["F6"]["p_disp_val"]),
        g=dict(c=R["G2"]["c_g"], crecimiento=P.g, tau=P.tau, eps=P.eps, conf=P.conf,
               modelo=R["G1"]["modelo_g"], validacion=R["G1"]["valid_g"].to_dict(),
               tabla=R["G3"]["g2"].replace({np.nan: None}).to_dict("index"), bootstrap=R["G3"]["conteo_c"]))
    resumen_final = a_nativo(resumen_final)
    (ruta / "resumen_consolidado.json").write_text(json.dumps(resumen_final, ensure_ascii=False, indent=2, allow_nan=False),
                                                   encoding="utf8")
    return resumen_final
