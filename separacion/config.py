"""Parámetros del grupo derivados de S (sección 3 del enunciado) y rutas de datos y resultados."""
from dataclasses import dataclass
from pathlib import Path

from scipy import stats


@dataclass(frozen=True)
class Parametros:
    """Parámetros personalizados del grupo. Todos se derivan de S = C1 + C2."""
    S: int
    tau: int          # umbral de latencia del SLO (ms)
    eps: float        # proporción máxima tolerada con L > tau
    conf: float       # nivel de confianza de todos los intervalos
    alpha: float      # probabilidad de error de cada intervalo (1 − conf)
    z_conf: float     # cuantil normal z_{1−alpha/2}
    g: float          # factor de crecimiento del tráfico (Parte G)
    t1: int           # corte entre imágenes pequeñas y medianas (KB)
    t2: int           # corte entre imágenes medianas y grandes (KB)
    pi_prod: float    # prevalencia de incidentes en producción (Parte E)
    kmax: int         # máximo de falsas alarmas diarias toleradas (E4)
    r: int            # desplazamiento del umbral k en B4
    N: int            # solicitudes por bloque en C4

    def __str__(self):
        return (f"S = {self.S}\n"
                f"tau = {self.tau} ms | eps = {self.eps:.2f} | conf = {self.conf:.2f} | g = {self.g} | "
                f"t1 = {self.t1} KB | t2 = {self.t2} KB\n"
                f"pi = {self.pi_prod} | kmax = {self.kmax} | r = {self.r} | N = {self.N}")


def parametros_del_grupo(S):
    """Calcula los parámetros de la sección 3 del enunciado a partir de la semilla S."""
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
    """Localiza la carpeta de los CSV y crea las carpetas de resultados.
    Devuelve (ruta_datos, ruta_resultados, ruta_resultados_G)."""
    candidatas = [Path("datos"), Path("."), Path(".."), Path("/content")]
    ruta_datos = next(p for p in candidatas if (p / "solicitudes.csv").exists())
    ruta_resultados = Path("resultados")
    ruta_g = ruta_resultados / "G"
    ruta_g.mkdir(parents=True, exist_ok=True)
    print("Datos leídos de:", ruta_datos.resolve())
    print("Resultados en:  ", ruta_resultados.resolve())
    return ruta_datos, ruta_resultados, ruta_g
