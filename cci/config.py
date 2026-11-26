"""Parámetros del grupo (sección 3 del enunciado) y rutas del proyecto."""
from dataclasses import dataclass
from pathlib import Path

from scipy import stats


@dataclass(frozen=True)
class Parametros:
    S: int
    tau: int          # umbral de latencia (ms)
    eps: float        # máximo permitido de P(L > tau)
    conf: float       # nivel de confianza
    alpha: float
    z_conf: float     # cuantil normal para conf
    g: float          # crecimiento del tráfico
    t1: int           # corte pequeña / mediana (KB)
    t2: int           # corte mediana / grande (KB)
    pi_prod: float    # prevalencia de incidentes en producción
    kmax: int         # máximo de falsas alarmas por día
    r: int            # se suma al umbral k de B4
    N: int            # tamaño del bloque de C4

    def __str__(self):
        return (f"S = {self.S}\n"
                f"tau = {self.tau} ms, eps = {self.eps:.2f}, conf = {self.conf:.2f}, g = {self.g}\n"
                f"t1 = {self.t1} KB, t2 = {self.t2} KB, pi = {self.pi_prod}, kmax = {self.kmax}, r = {self.r}, N = {self.N}")


def parametros_del_grupo(S):
    conf = {0: 0.90, 1: 0.95, 2: 0.99}[S % 3]
    alpha = 1 - conf
    return Parametros(
        S=S,
        tau=180 + 10 * (S % 9),
        eps=0.01 * (1 + S % 3),
        conf=conf,
        alpha=alpha,
        z_conf=stats.norm.ppf(1 - alpha / 2),
        g=round(1.4 + 0.1 * (S % 7), 1),
        t1=80 + 10 * (S % 5),
        t2=400 + 25 * (S % 5),
        pi_prod=(4 + S % 6) / 1000,
        kmax=4 + S % 5,
        r=S % 6,
        N=10 + S % 11,
    )


def preparar_rutas():
    """Busca la carpeta de los CSV y crea las carpetas de resultados."""
    candidatas = [Path("datos"), Path("."), Path(".."), Path("/content")]
    ruta_datos = next(p for p in candidatas if (p / "solicitudes.csv").exists())
    ruta_resultados = Path("resultados")
    ruta_g = ruta_resultados / "G"
    ruta_g.mkdir(parents=True, exist_ok=True)
    (ruta_resultados / "tablas").mkdir(exist_ok=True)
    return ruta_datos, ruta_resultados, ruta_g


def guardar_tabla(tabla, nombre, ruta_resultados):
    """Guarda una tabla completa en resultados/tablas/ para usarla en el informe."""
    tabla.to_csv(ruta_resultados / "tablas" / f"{nombre}.csv")
