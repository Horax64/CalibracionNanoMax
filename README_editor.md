# Editor de trayectorias

Herramienta visual para **graficar, recortar y limpiar** los `.csv` de
trayectorias antes de usarlos en la calibración. Es el segundo paso del
flujo de trabajo:

```
trackeo_particulas.py  →  editor_trayectorias.py  →  Calibración/calibracion_v2.py
      (trackear)              (pulir datos)              (ajustes y promedios)
```

**Archivo**: `editor_trayectorias.py`.

## Requisitos

Los mismos del trackeo (`numpy`, `pandas`, `matplotlib`). Acepta cualquier
csv con columnas `X` e `Y`: los crudos del tracker, los `_recortado` de una
edición anterior, o los `_proc` del formato viejo.

## Uso

```
py editor_trayectorias.py              (abre un diálogo para elegir el .csv)
py editor_trayectorias.py datos.csv    (o directo por argumento)
```

Se abre una ventana con la trayectoria y una fila de botones.

## Controles

- **Recortar**: arrastrar un rectángulo sobre el gráfico (botón izquierdo
  **o derecho** del mouse) **elimina los puntos de adentro**. Sirve para
  recortar en X, en Y o cualquier zona.
- **Lupa/mano del toolbar**: funcionan normal para navegar. Con la lupa
  activa, el botón izquierdo hace zoom y el **derecho sigue eliminando** —
  no hace falta cambiar de herramienta. El zoom se conserva al eliminar.
- **[Vista]**: alterna Trayectoria XY → X vs tiempo → Y vs tiempo. En las
  vistas temporales el rectángulo permite recortar rangos de tiempo
  (por ejemplo, sacar un tramo entero del barrido).
- **[Deshacer]**: revierte el último recorte/limpieza, tantas veces como
  haga falta.
- **[Marcar outliers]**: detecta picos aislados y los pinta de rojo, sin
  tocarlos. El criterio: un punto lejos de sus dos vecinos temporales
  mientras los vecinos son consistentes entre sí — así los pasos del
  barrido no dan falsa alarma.
- **[Quitar outliers]**: elimina los marcados.
- **[Quitar perdidos]**: elimina los frames con `estado='perdido'` (los
  transitorios de flyback que el tracker ya marcó como no confiables).
  Recomendado apretarlo siempre como primer paso.
- **[Guardar]**: escribe **`<nombre>_recortado.csv`** junto al original.
  **Nunca pisa el archivo original.** Se conservan todas las columnas.

La consola informa cada acción (cuántos puntos se eliminaron, dónde se
guardó).

## Flujo recomendado

1. Abrir el csv del trackeo.
2. `Quitar perdidos`.
3. Lupa para acercarse a las zonas dudosas → botón derecho para recortar.
4. `Marcar outliers` (con lo bien que funciona el tracker, normalmente da 0).
5. `Guardar` → usar el `_recortado.csv` en la calibración.

## Ajustes

Los umbrales del detector de outliers son constantes comentadas al inicio
del script (`VECINOS_CERCA`, `FACTOR_VECINOS`, `SALTO_OUTLIER`,
`FACTOR_SALTO`) por si alguna vez hay que afinarlos.
