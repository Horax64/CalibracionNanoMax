# Trackeo automático de barridos

Trackea la trayectoria de la muestra de discos de silicio en los videos de
barrido, **sin selección manual de ROI**: detecta los discos solo, sigue el
movimiento global de la muestra y maneja los saltos de fila (flybacks) del
scanning. Es el primer paso del flujo de trabajo:

```
trackeo_particulas.py  →  editor_trayectorias.py  →  Calibración/calibracion_v2.py
      (trackear)              (pulir datos)              (ajustes y promedios)
```

**Archivos**: `trackeo_particulas.py` (script principal) + `trackerclass_v5.py`
(el motor de trackeo; no se corre directo).

## Requisitos

Python 3 con:

```
pip install opencv-python numpy pandas matplotlib
```

## Uso

```
py trackeo_particulas.py                        (abre un diálogo para elegir el video)
py trackeo_particulas.py video.mp4              (o directo por argumento)
py trackeo_particulas.py video.mp4 --sin-preview --sin-plot   (modo silencioso)
```

Flujo: se elige el video → se muestra una ventana de preview con los discos
trackeados (tecla `q` o `Esc` para abortar guardando lo hecho) → al terminar
se guardan los datos y se muestra el gráfico de trayectorias.

La ventana de preview se puede desactivar con `--sin-preview` (el trackeo es
un poco más rápido). Velocidad típica: 50–70 fps de procesamiento a 1080p.

## Salidas

En `Datos_tray/Datos_crudos/`:

- **`<video>.csv`** — una fila por frame, columnas:
  - `X`, `Y`: posición de la muestra en píxeles (de un punto de referencia
    fijo a la muestra; solo importan los desplazamientos relativos).
  - `Frame`, `Tiempo_seg`: tiempo.
  - `confianza`: correlación mediana del ensamble (0–1).
  - `n_confiables`: cuántos de los ~20 discos del ensamble midieron bien.
  - `estado`: `ok` (dato bueno), `perdido` (transitorio de flyback, posición
    congelada, **descartar para análisis**), `relocalizado` (primer frame
    después de un salto de fila; el dato es válido).
- **`<video>_trayectoria.png`** — figura con la trayectoria XY, X(t), Y(t) y
  la calidad del trackeo.

## Cómo funciona (resumen)

- La muestra es una red periódica de discos que se mueve rígida. En vez de
  seguir un disco, se sigue un **ensamble de ~20 discos** por correlación
  (NCC) y el desplazamiento global es la mediana robusta de todos.
- Todo se **autocalibra** midiendo los vectores de la red por autocorrelación
  2D — funciona aunque la muestra esté rotada respecto de la cámara.
- En los flybacks la correspondencia es ambigua módulo el período de la red
  (~105 px): se resuelve prediciendo el aterrizaje (el piezo repite el
  comando) y anclando la fase de la red detectada, con confirmación en
  varios frames. Un watchdog verifica periódicamente que el ensamble siga
  sincronizado con la red real (y no con el polvo fijo a la cámara).

## Consejos de grabación

- **Empezar a grabar con la muestra quieta, al inicio del barrido.** Si el
  video arranca a mitad de una fila, la posición absoluta en X de la primera
  fila puede quedar corrida un múltiplo del período respecto de las demás
  (el script lo advierte; los pasos dentro de cada fila no se afectan).
- Iluminación y foco razonables: el trackeo tolera bastante, pero necesita
  ver los discos (los videos con contraste casi nulo no funcionan).

## Limitaciones conocidas

- Los frames de flyback quedan como `estado='perdido'` — es lo esperado,
  filtrarlos (el editor tiene un botón para eso).
- En barridos tipo grilla con mucho wobble de cross-talk en los retornos,
  alguna fila aislada puede quedar corrida un período (~105 px) en X
  absoluto. Los datos internos de esa fila siguen siendo correctos.
- Los "aterrizajes" que imprime el log incluyen el jitter del momento de
  relocalización — no usarlos como métrica de repetibilidad del piezo.

## Parámetros útiles (constructor de `TrackerBarridos`)

| Parámetro | Default | Qué hace |
|---|---|---|
| `canal` | 0 (azul) | Canal de color usado para trackear |
| `n_trackers` | 20 | Tamaño del ensamble de discos |
| `t_inicio` | 0.0 | Segundo del video donde empezar |
| `preview_cada` | 3 | Dibuja 1 de cada N frames en el preview |
