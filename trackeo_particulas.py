# -*- coding: utf-8 -*-
"""
Trackeo automático de barridos — flujo completo.

Uso:  py trackeo_particulas.py [video.mp4] [--sin-preview] [--sin-plot]

Sin argumentos abre un diálogo para elegir el video. El trackeo es totalmente
automático (sin selección manual de ROI): detecta los discos, sigue el
desplazamiento global de la muestra, maneja los flybacks del scanning y al
terminar guarda el CSV en Datos_tray/Datos_crudos y muestra las trayectorias.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from trackerclass_v5 import TrackerBarridos

CARPETA_SCRIPT = Path(__file__).resolve().parent
CARPETA_SALIDA = CARPETA_SCRIPT / "Datos_tray" / "Datos_crudos"
CARPETA_VIDEOS = CARPETA_SCRIPT / "Videos"


def elegir_video():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return Path(args[0])
    from tkinter import Tk, filedialog
    raiz = Tk()
    raiz.withdraw()
    ruta = filedialog.askopenfilename(
        title="Abrir video a trackear",
        initialdir=CARPETA_VIDEOS if CARPETA_VIDEOS.exists() else Path.home(),
        filetypes=[("Videos", "*.mp4 *.avi *.mov"), ("Todos", "*.*")])
    raiz.destroy()
    return Path(ruta) if ruta else None


def graficar(df, nombre, ruta_png, mostrar=True):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ok = df["estado"] == "ok"
    reloc = df["estado"] == "relocalizado"
    perdido = df["estado"] == "perdido"

    ax = axes[0, 0]
    ax.scatter(df.loc[ok, "X"], df.loc[ok, "Y"], s=4, c="tab:blue", label="trackeado")
    if perdido.any():
        ax.scatter(df.loc[perdido, "X"], df.loc[perdido, "Y"], s=4, c="0.7", label="perdido")
    if reloc.any():
        ax.scatter(df.loc[reloc, "X"], df.loc[reloc, "Y"], s=30, c="tab:red",
                   marker="x", label="relocalización")
    ax.invert_yaxis()
    ax.set_xlabel("X (píxeles)")
    ax.set_ylabel("Y (píxeles)")
    ax.set_title("Trayectoria del barrido")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    for ax, col in ((axes[0, 1], "X"), (axes[1, 0], "Y")):
        ax.plot(df["Tiempo_seg"], df[col], lw=0.8)
        if reloc.any():
            for t in df.loc[reloc, "Tiempo_seg"]:
                ax.axvline(t, color="tab:red", lw=0.6, alpha=0.6)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel(f"{col} (píxeles)")
        ax.set_title(f"{col} vs tiempo")
        ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    ax.plot(df["Tiempo_seg"], df["confianza"], lw=0.6, label="confianza NCC")
    ax.plot(df["Tiempo_seg"], df["n_confiables"] / max(df["n_confiables"].max(), 1),
            lw=0.6, alpha=0.7, label="fracción de discos confiables")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Calidad del trackeo")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(nombre)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    print(f"[i] Gráfico guardado en {ruta_png}")
    if mostrar:
        plt.show()
    plt.close(fig)


def main():
    ruta_video = elegir_video()
    if not ruta_video or not ruta_video.exists():
        print("No se seleccionó ningún video. Saliendo.")
        return

    mostrar_preview = "--sin-preview" not in sys.argv
    mostrar_plot = "--sin-plot" not in sys.argv
    if not mostrar_plot:
        matplotlib.use("Agg")

    print(f"Trackeando: {ruta_video.name}")
    tracker = TrackerBarridos(ruta_video, mostrar_video=mostrar_preview)
    res = tracker.track()

    df = pd.DataFrame({k: res[k] for k in
                       ("X", "Y", "Frame", "Tiempo_seg", "confianza", "n_confiables", "estado")})

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_csv = CARPETA_SALIDA / f"{ruta_video.stem}.csv"
    df.to_csv(ruta_csv, index=False)
    print(f"[i] Datos guardados en {ruta_csv}")

    # --- resumen de control de calidad ---
    frac_ok = (df["estado"] == "ok").mean()
    print(f"\n=== Resumen ===")
    print(f"Frames trackeados: {len(df)}  |  ok: {100 * frac_ok:.1f}%  |  "
          f"relocalizaciones: {len(res['relocalizaciones'])}  |  "
          f"velocidad: {res['fps_procesamiento']:.0f} fps")
    if res["relocalizaciones"]:
        print("Saltos de fila detectados en los frames: "
              f"{[f for f, *_ in res['relocalizaciones']]}")
        print("(si el video no empieza al inicio de una fila, el X absoluto de las filas\n"
              " posteriores a la primera puede quedar corrido en un múltiplo del período;\n"
              " los pasos dentro de cada fila y los saltos en Y no se ven afectados)")
    for adv in res["advertencias"]:
        print(f"[!] {adv}")

    graficar(df, ruta_video.stem, CARPETA_SALIDA / f"{ruta_video.stem}_trayectoria.png",
             mostrar=mostrar_plot)


if __name__ == "__main__":
    main()
