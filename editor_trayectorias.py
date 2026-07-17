# -*- coding: utf-8 -*-
"""
Editor de trayectorias — graficar, recortar y limpiar los .csv del trackeo.

Uso:  py editor_trayectorias.py [datos.csv]

Sin argumentos abre un diálogo para elegir el .csv (funciona con los crudos
del tracker y con cualquier csv que tenga columnas X e Y).

Controles:
  - Arrastrar un rectángulo sobre el gráfico ELIMINA los puntos de adentro
    (botón izquierdo o derecho). Sirve para recortar en X, en Y o cualquier
    zona; en las vistas "X vs t" e "Y vs t" permite recortar por tiempo.
  - La lupa y la mano del toolbar funcionan normalmente para navegar: si
    la lupa está activa, el botón izquierdo hace zoom y el DERECHO sigue
    eliminando. El zoom se conserva al eliminar puntos.
  - [Vista]            alterna entre trayectoria XY, X vs t, Y vs t.
  - [Deshacer]         revierte el último recorte/limpieza (todas las veces
                        que haga falta).
  - [Marcar outliers]  detecta picos aislados (punto lejos de sus dos vecinos
                        temporales mientras los vecinos son consistentes
                        entre sí) y los pinta de rojo, sin eliminarlos.
  - [Quitar outliers]  elimina los outliers marcados.
  - [Quitar perdidos]  elimina los frames con estado 'perdido' (transitorios
                        de flyback que el tracker ya marcó como no confiables).
  - [Guardar]          escribe <nombre>_recortado.csv junto al original
                        (nunca pisa el archivo original).

Los umbrales del detector de outliers se pueden ajustar acá abajo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

# --- parámetros del detector de outliers -------------------------------- #
# un punto es outlier si sus dos vecinos temporales están cerca entre sí
# (movimiento local < VECINOS_CERCA px o < FACTOR_VECINOS x el paso típico)
# pero el punto está lejos de ambos (> SALTO_OUTLIER px y > FACTOR_SALTO x paso)
VECINOS_CERCA = 3.0
FACTOR_VECINOS = 2.0
SALTO_OUTLIER = 2.0
FACTOR_SALTO = 5.0

CARPETA_SCRIPT = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_SCRIPT / "Datos_tray" / "Datos_crudos"


def elegir_csv():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return Path(args[0])
    from tkinter import Tk, filedialog
    raiz = Tk()
    raiz.withdraw()
    ruta = filedialog.askopenfilename(
        title="Abrir trayectoria a editar",
        initialdir=CARPETA_DATOS if CARPETA_DATOS.exists() else Path.home(),
        filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
    raiz.destroy()
    return Path(ruta) if ruta else None


def detectar_outliers(x, y, orden):
    """Picos aislados: lejos de ambos vecinos temporales, con vecinos
    consistentes entre sí. No marca los pasos del barrido (ahí los vecinos
    difieren entre sí) ni el ruido normal de dwell."""
    n = len(x)
    marca = np.zeros(n, dtype=bool)
    if n < 3:
        return marca
    xs, ys = x[orden], y[orden]
    d_priv = np.hypot(np.diff(xs), np.diff(ys))          # punto i vs i-1
    paso_tipico = max(np.median(d_priv), 0.05)
    d_salt = np.hypot(xs[2:] - xs[:-2], ys[2:] - ys[:-2])  # vecino a vecino
    cerca = max(VECINOS_CERCA, FACTOR_VECINOS * paso_tipico)
    lejos = max(SALTO_OUTLIER, FACTOR_SALTO * paso_tipico, 1.5 * cerca)
    es_pico = (d_salt < cerca) & (d_priv[:-1] > lejos) & (d_priv[1:] > lejos)
    marca_ordenada = np.zeros(n, dtype=bool)
    marca_ordenada[1:-1] = es_pico
    marca[orden] = marca_ordenada
    return marca


class EditorTrayectorias:
    VISTAS = ("Trayectoria XY", "X vs tiempo", "Y vs tiempo")

    def __init__(self, ruta_csv):
        self.ruta = Path(ruta_csv)
        self.df = pd.read_csv(self.ruta)
        if not {"X", "Y"}.issubset(self.df.columns):
            raise ValueError("El csv debe tener columnas X e Y.")
        n = len(self.df)
        if "Tiempo_seg" in self.df.columns:
            self.t = self.df["Tiempo_seg"].values
            self.etiqueta_t = "Tiempo (s)"
        elif "Frame" in self.df.columns:
            self.t = self.df["Frame"].values.astype(float)
            self.etiqueta_t = "Frame"
        else:
            self.t = np.arange(n, dtype=float)
            self.etiqueta_t = "índice"

        self.activos = np.ones(n, dtype=bool)   # puntos vivos
        self.outliers = np.zeros(n, dtype=bool)  # marcados en rojo
        self.historial = []                       # pila para deshacer
        self.vista = 0

        self.fig, self.ax = plt.subplots(figsize=(11, 7.5))
        self.fig.canvas.manager.set_window_title(f"Editor de trayectorias — {self.ruta.name}")
        self.fig.subplots_adjust(bottom=0.17)

        # --- botones ---
        defs = [("Vista", self.cambiar_vista),
                ("Deshacer", self.deshacer),
                ("Marcar\noutliers", self.marcar_outliers),
                ("Quitar\noutliers", self.quitar_outliers),
                ("Quitar\nperdidos", self.quitar_perdidos),
                ("Guardar", self.guardar)]
        self.botones = []
        ancho, sep, x0 = 0.115, 0.018, 0.08
        for k, (texto, accion) in enumerate(defs):
            eje = self.fig.add_axes([x0 + k * (ancho + sep), 0.03, ancho, 0.075])
            b = Button(eje, texto)
            b.on_clicked(accion)
            self.botones.append(b)

        self.selector = None
        self.redibujar()

    def _crear_selector(self):
        """(Re)crea el selector: ax.clear() lo rompe, hay que rearmarlo tras
        cada redibujado o arrastrar deja de eliminar puntos."""
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
        self.selector = RectangleSelector(
            self.ax, self.recorte_rectangulo, useblit=True, button=[1, 3],
            minspanx=3, minspany=3, spancoords="pixels", interactive=False)

    # ------------------------------------------------------------------ #
    def coords_vista(self):
        """(x, y) de cada punto según la vista actual."""
        if self.vista == 0:
            return self.df["X"].values, self.df["Y"].values
        if self.vista == 1:
            return self.t, self.df["X"].values
        return self.t, self.df["Y"].values

    def redibujar(self, mantener_zoom=False):
        lims = (self.ax.get_xlim(), self.ax.get_ylim()) if mantener_zoom else None
        self.ax.clear()
        cx, cy = self.coords_vista()
        act, out = self.activos, self.outliers
        normales = act & ~out
        self.ax.scatter(cx[normales], cy[normales], s=6, c="tab:blue",
                        label=f"activos ({act.sum()})")
        if (act & out).any():
            self.ax.scatter(cx[act & out], cy[act & out], s=22, c="tab:red",
                            marker="x", label=f"outliers ({(act & out).sum()})")
        eliminados = (~act).sum()
        if self.vista == 0:
            self.ax.set_xlabel("X (píxeles)")
            self.ax.set_ylabel("Y (píxeles)")
        else:
            self.ax.set_xlabel(self.etiqueta_t)
            self.ax.set_ylabel(("X" if self.vista == 1 else "Y") + " (píxeles)")
        if lims is not None:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        elif self.vista == 0:
            self.ax.invert_yaxis()
        self.ax.set_title(f"{self.VISTAS[self.vista]}  |  "
                          f"{act.sum()} puntos, {eliminados} eliminados  |  "
                          "arrastrar rectángulo (izq. o der.) = eliminar")
        self.ax.legend(loc="best")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self._crear_selector()
        self.fig.canvas.draw_idle()

    def empujar_historial(self):
        self.historial.append(self.activos.copy())
        if len(self.historial) > 60:
            self.historial.pop(0)

    # ------------------------------------------------------------------ #
    #  Acciones                                                          #
    # ------------------------------------------------------------------ #
    def recorte_rectangulo(self, ini, fin):
        # si la lupa/mano del toolbar está activa, el botón izquierdo es de
        # ella (zoom/pan); solo el derecho elimina en ese caso
        barra = getattr(self.fig.canvas, "toolbar", None)
        if barra is not None and getattr(barra, "mode", "") and ini.button == 1:
            return
        x0, x1 = sorted((ini.xdata, fin.xdata))
        y0, y1 = sorted((ini.ydata, fin.ydata))
        cx, cy = self.coords_vista()
        adentro = self.activos & (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        if adentro.any():
            self.empujar_historial()
            self.activos[adentro] = False
            print(f"[i] Eliminados {adentro.sum()} puntos "
                  f"(quedan {self.activos.sum()}). 'Deshacer' los recupera.")
            self.redibujar(mantener_zoom=True)

    def cambiar_vista(self, _evt):
        self.vista = (self.vista + 1) % 3
        self.redibujar()

    def deshacer(self, _evt):
        if self.historial:
            self.activos = self.historial.pop()
            print(f"[i] Deshecho (quedan {self.activos.sum()} puntos).")
            self.redibujar(mantener_zoom=True)
        else:
            print("[i] No hay nada para deshacer.")

    def marcar_outliers(self, _evt):
        idx = np.nonzero(self.activos)[0]
        x = self.df["X"].values[idx]
        y = self.df["Y"].values[idx]
        orden = np.argsort(self.t[idx], kind="stable")
        marca = detectar_outliers(x, y, orden)
        self.outliers[:] = False
        self.outliers[idx[marca]] = True
        print(f"[i] {marca.sum()} outliers marcados (en rojo). "
              "'Quitar outliers' los elimina.")
        self.redibujar(mantener_zoom=True)

    def quitar_outliers(self, _evt):
        objetivo = self.activos & self.outliers
        if objetivo.any():
            self.empujar_historial()
            self.activos[objetivo] = False
            self.outliers[:] = False
            print(f"[i] Eliminados {objetivo.sum()} outliers.")
            self.redibujar(mantener_zoom=True)
        else:
            print("[i] No hay outliers marcados (usá 'Marcar outliers' primero).")

    def quitar_perdidos(self, _evt):
        if "estado" not in self.df.columns:
            print("[i] Este csv no tiene columna 'estado' (formato viejo).")
            return
        objetivo = self.activos & (self.df["estado"].values == "perdido")
        if objetivo.any():
            self.empujar_historial()
            self.activos[objetivo] = False
            print(f"[i] Eliminados {objetivo.sum()} frames 'perdido'.")
            self.redibujar(mantener_zoom=True)
        else:
            print("[i] No quedan frames 'perdido'.")

    def guardar(self, _evt):
        salida = self.ruta.with_name(self.ruta.stem + "_recortado.csv")
        self.df[self.activos].to_csv(salida, index=False)
        print(f"[i] Guardado: {salida}  ({self.activos.sum()} puntos, "
              f"{(~self.activos).sum()} eliminados)")


def main():
    ruta = elegir_csv()
    if not ruta or not ruta.exists():
        print("No se seleccionó ningún archivo. Saliendo.")
        return
    print(__doc__)
    editor = EditorTrayectorias(ruta)
    plt.show()


if __name__ == "__main__":
    main()
