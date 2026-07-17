# -*- coding: utf-8 -*-
"""
Tracker de barridos v5 — trackeo global de la muestra de discos de silicio.

Idea central: la muestra es una red periódica de discos que se mueve rígida.
En lugar de seguir UN disco (ambiguo en una red periódica y frágil en los
flybacks), se sigue un ENSAMBLE de ~20 discos con correlación normalizada
(NCC) en ventanas chicas, y el desplazamiento global se toma como la mediana
robusta de los desplazamientos individuales. Todo se auto-calibra a partir
del período de la red medido por autocorrelación (sin ROI manual).

Hechos medidos en los videos que condicionan el diseño:
  - Período de la red: ~105 px, con rotación de ~1° respecto de la cámara
    (por eso se estiman los VECTORES de red a1, a2 y no solo los períodos).
  - La iluminación está fija a la cámara: la apariencia de un disco cambia
    al moverse la muestra -> re-templateo escalonado del ensamble.
  - El polvo/viñeteo está fijo a la cámara y no hay textura anclada a la
    muestra: tras un flyback la correspondencia es ambigua módulo la red.
    Se resuelve con predicción geométrica (inicio de fila + paso en y
    aprendido), que elige el nodo de red correcto.
  - El flyback salta cientos de px en 1-2 frames. Envuelto módulo el
    período se disfraza de paso normal: se detecta porque va contra la
    tendencia de la fila o supera el paso máximo legítimo. La medición
    durante el salto sigue siendo válida módulo la red: si el match es
    nítido, la relocalización preferida es "ajustar al nodo" la propia
    medición NCC, usando como predicción el aterrizaje del flyback
    anterior más el paso de fila aprendido (el piezo repite el comando).
"""
import time

import cv2 as cv
import numpy as np


def _wrap(v, P):
    """Envuelve v al intervalo [-P/2, P/2)."""
    return (v + P / 2.0) % P - P / 2.0


def _mediana_circular(vals, P):
    """Mediana de fases módulo P (asume cluster compacto)."""
    vals = np.asarray(vals, dtype=np.float64)
    ref = vals[0]
    return (ref + np.median(_wrap(vals - ref, P))) % P


def _subpixel_parabola(mapa, px, py):
    """Refinamiento subpíxel del máximo entero (px, py) por parábola 3x3."""
    h, w = mapa.shape
    if px <= 0 or px >= w - 1 or py <= 0 or py >= h - 1:
        return float(px), float(py)
    c = mapa[py, px]
    den_x = 2.0 * (mapa[py, px - 1] - 2.0 * c + mapa[py, px + 1])
    den_y = 2.0 * (mapa[py - 1, px] - 2.0 * c + mapa[py + 1, px])
    dx = (mapa[py, px - 1] - mapa[py, px + 1]) / den_x if den_x != 0 else 0.0
    dy = (mapa[py - 1, px] - mapa[py + 1, px]) / den_y if den_y != 0 else 0.0
    # Un desplazamiento mayor a medio píxel indica ajuste degenerado
    dx = dx if abs(dx) <= 0.6 else 0.0
    dy = dy if abs(dy) <= 0.6 else 0.0
    return px + dx, py + dy


class TrackerBarridos:
    """Trackea la trayectoria global de la muestra a lo largo del video."""

    def __init__(self, video_path, canal=0, n_trackers=20, t_inicio=0.0,
                 mostrar_video=True, preview_cada=3, verbose=True, debug_rango=None):
        self.debug_rango = debug_rango  # (a, b): imprime diagnóstico por frame
        self.video_path = str(video_path)
        self.canal = canal
        self.n_trackers = n_trackers
        self.mostrar_video = mostrar_video
        self.preview_cada = max(1, preview_cada)
        self.verbose = verbose

        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {self.video_path}")
        self.fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        cap.release()
        self.frame_inicio = int(round(t_inicio * self.fps))

        self.advertencias = []
        self.relocalizaciones = []  # (frame_abs, motivo, S_previo, S_nuevo)

    # ------------------------------------------------------------------ #
    #  Utilidades de imagen                                              #
    # ------------------------------------------------------------------ #
    def _gris(self, frame_bgr):
        """Canal de trabajo (por defecto azul, el de mayor señal con esta luz)."""
        if self.canal in (0, 1, 2):
            return frame_bgr[:, :, self.canal]
        return cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    def _estimar_periodo(self, gris):
        """Período (Px, Py) de la red por autocorrelación de un recorte pasabanda."""
        h, w = gris.shape
        cy, cx = h // 2, w // 2
        hh, hw = min(256, cy - 8), min(512, cx - 8)
        crop = gris[cy - hh:cy + hh, cx - hw:cx + hw].astype(np.float32)
        hp = crop - cv.GaussianBlur(crop, (0, 0), 15)
        hp *= np.outer(np.hanning(hp.shape[0]), np.hanning(hp.shape[1])).astype(np.float32)
        F = np.fft.rfft2(hp)
        ac = np.fft.irfft2(F * np.conj(F))

        def primer_pico(perfil):
            for i in range(8, len(perfil) - 1):
                if perfil[i] > perfil[i - 1] and perfil[i] >= perfil[i + 1] and perfil[i] > 0.15 * perfil[0]:
                    return i
            return -1

        Px = primer_pico(ac[0, :ac.shape[1] // 2])
        Py = primer_pico(ac[:ac.shape[0] // 2, 0])
        if not (20 < Px < 400):
            self.advertencias.append(f"Período X no detectado (autocorr dio {Px}); uso 100 px.")
            Px = 100
        if not (20 < Py < 400):
            self.advertencias.append(f"Período Y no detectado (autocorr dio {Py}); uso 100 px.")
            Py = 100
        return float(Px), float(Py)

    def _estimar_red_autocorr(self, gris):
        """Vectores de red a1, a2 por autocorrelación 2D: soporta rotación
        arbitraria de la muestra (los picos laterales de la autocorrelación
        están en los vectores de red, sin importar el ángulo).

        Devuelve la base B = [a1 a2] (columnas) o None si no hay red clara.
        """
        h, w = gris.shape
        cy, cx = h // 2, w // 2
        hh, hw = min(256, cy - 8), min(512, cx - 8)
        crop = gris[cy - hh:cy + hh, cx - hw:cx + hw].astype(np.float32)
        hp = crop - cv.GaussianBlur(crop, (0, 0), 15)
        hp *= np.outer(np.hanning(hp.shape[0]), np.hanning(hp.shape[1])).astype(np.float32)
        F = np.fft.rfft2(hp)
        ac = np.fft.fftshift(np.fft.irfft2(F * np.conj(F))).astype(np.float32)
        c0y, c0x = ac.shape[0] // 2, ac.shape[1] // 2
        pico0 = float(ac[c0y, c0x])

        mx = cv.dilate(ac, np.ones((15, 15), np.uint8))
        ys, xs = np.nonzero((ac >= mx) & (ac > 0.10 * pico0))
        vx, vy = xs - c0x, ys - c0y
        r = np.hypot(vx, vy)
        m = (r > 18) & (r < 380)
        if m.sum() < 2:
            return None
        vx, vy, r = vx[m], vy[m], r[m]
        orden = np.argsort(r)

        def subpix(j):
            sx, sy = _subpixel_parabola(ac, int(vx[j] + c0x), int(vy[j] + c0y))
            return np.array([sx - c0x, sy - c0y])

        a1 = subpix(orden[0])
        a2 = None
        for j in orden[1:]:
            v = np.array([vx[j], vy[j]], dtype=float)
            cosang = abs(v @ a1) / (np.linalg.norm(v) * np.linalg.norm(a1) + 1e-9)
            if cosang < 0.7:  # no colineal con a1
                a2 = subpix(j)
                break
        if a2 is None:
            return None
        # convención: a1 apunta hacia +x, a2 hacia +y
        if abs(a1[0]) < abs(a2[0]):
            a1, a2 = a2, a1
        if a1[0] < 0:
            a1 = -a1
        if a2[1] < 0:
            a2 = -a2
        if not (18 < np.linalg.norm(a1) < 400 and 18 < np.linalg.norm(a2) < 400):
            return None
        return np.column_stack([a1, a2])

    def _refinar_base(self, discos, base):
        """Refina a1, a2 con las diferencias entre discos vecinos detectados."""
        if len(discos) < 20:
            return base
        refinada = base.copy()
        for col in (0, 1):
            a = base[:, col]
            tol = 0.25 * np.linalg.norm(a)
            difs = []
            for d in discos:
                v = discos - d
                cerca = v[np.hypot(*(v - a).T) < tol]
                if len(cerca):
                    difs.append(cerca[np.argmin(np.hypot(*(cerca - a).T))])
            if len(difs) >= 10:
                refinada[:, col] = np.median(difs, axis=0)
        return refinada

    def _mapa_discos(self, gris_f32):
        """Respuesta DoG a la ESCALA DEL DISCO (en valor absoluto: discos claros
        u oscuros). La escala grande es clave: el polvo fijo a la cámara (motas
        de pocos px) casi no responde, y un disco borroneado por movimiento
        rápido (flyback) tampoco — eso evita anclarse al patrón fijo."""
        banda = cv.GaussianBlur(gris_f32, (0, 0), self.sigma_blob) - \
                cv.GaussianBlur(gris_f32, (0, 0), 2.0 * self.sigma_blob)
        return np.abs(banda)

    def _detectar_discos(self, gris, region=None, n_max=60, min_dist=None, margen=None,
                         umbral_min=None):
        """Centros de discos (subpíxel) por máximos locales de la respuesta DoG.

        region: (x0, y0, x1, y1) para restringir la búsqueda. umbral_min es un
        umbral ABSOLUTO de respuesta (referido a self.resp_ref): filtra frames
        borroneados y polvo. Devuelve (posiciones (N,2), respuestas (N,)),
        ordenadas por respuesta descendente y con separación mínima min_dist.
        """
        P = self.periodo_med
        if min_dist is None:
            min_dist = 0.7 * P
        if margen is None:
            margen = self.margen_seguro
        x0, y0, x1, y1 = region if region is not None else (0, 0, self.ancho, self.alto)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.ancho, int(x1)), min(self.alto, int(y1))
        crop = np.ascontiguousarray(gris[y0:y1, x0:x1]).astype(np.float32)
        vacio = (np.empty((0, 2)), np.empty(0))
        if crop.shape[0] < 6 * self.sigma_blob or crop.shape[1] < 6 * self.sigma_blob:
            return vacio

        resp = self._mapa_discos(crop)
        k = int(0.5 * P) | 1
        maximo_local = cv.dilate(resp, np.ones((k, k), np.uint8))
        umbral = max(np.percentile(resp, 90), 0.25 * resp.max())
        if umbral_min is not None:
            umbral = max(umbral, umbral_min)
        ys, xs = np.nonzero((resp >= maximo_local) & (resp > umbral))
        if len(xs) == 0:
            return vacio
        vals = resp[ys, xs]
        orden = np.argsort(vals)[::-1]

        elegidos, respuestas = [], []
        for idx in orden:
            x, y = float(xs[idx]), float(ys[idx])
            sx, sy = _subpixel_parabola(resp, int(x), int(y))
            ax, ay = x0 + sx, y0 + sy  # coordenadas absolutas
            if not (margen < ax < self.ancho - margen and margen < ay < self.alto - margen):
                continue
            if any((ax - ex) ** 2 + (ay - ey) ** 2 < min_dist ** 2 for ex, ey in elegidos):
                continue
            elegidos.append((ax, ay))
            respuestas.append(vals[idx])
            if len(elegidos) >= n_max:
                break
        return np.array(elegidos), np.array(respuestas)

    # ------------------------------------------------------------------ #
    #  Trackers individuales                                             #
    # ------------------------------------------------------------------ #
    def _capturar_template(self, gris, pos_xy, S_actual, umbral_retemplate):
        """Crea un tracker: template centrado en la posición entera más cercana."""
        st2 = self.semitemplate
        cx, cy = int(round(pos_xy[0])), int(round(pos_xy[1]))
        if not (st2 <= cx < self.ancho - st2 and st2 <= cy < self.alto - st2):
            return None
        template = np.ascontiguousarray(gris[cy - st2:cy + st2, cx - st2:cx + st2])
        return {
            "template": template,
            "anclaje": np.array([cx, cy], dtype=np.float64),  # centro del contenido al capturar
            "S_cap": S_actual.copy(),
            "pos": np.array([cx, cy], dtype=np.float64),      # última posición medida
            "umbral_ret": umbral_retemplate,
            "conf": 1.0,
            "malos": 0,
        }

    def _sembrar_trackers(self, gris, S_actual, posiciones=None):
        """(Re)inicializa el ensamble completo a partir de discos detectados."""
        if posiciones is None or len(posiciones) < 6:
            posiciones, _ = self._detectar_discos(
                gris, n_max=self.n_trackers, umbral_min=0.55 * self.resp_ref)
        umbrales = np.linspace(0.45, 0.85, max(2, self.n_trackers)) * self.radio_busqueda
        trackers = []
        for i, pos in enumerate(posiciones[: self.n_trackers]):
            t = self._capturar_template(gris, pos, S_actual, umbrales[i % len(umbrales)])
            if t is not None:
                trackers.append(t)
        return trackers

    def _medir_tracker(self, gris, t, S_pred):
        """NCC del template en la ventana predicha. Devuelve (pos_medida, conf) o None."""
        st2, r = self.semitemplate, self.radio_busqueda
        esperado = t["anclaje"] + (S_pred - t["S_cap"])
        ex, ey = int(round(esperado[0])), int(round(esperado[1]))
        x0, x1 = ex - st2 - r, ex + st2 + r
        y0, y1 = ey - st2 - r, ey + st2 + r
        if x0 < 0 or y0 < 0 or x1 > self.ancho or y1 > self.alto:
            return None  # ventana fuera del frame: el disco salió -> respawn
        ventana = np.ascontiguousarray(gris[y0:y1, x0:x1])
        res = cv.matchTemplate(ventana, t["template"], cv.TM_CCOEFF_NORMED)
        _, conf, _, (mx, my) = cv.minMaxLoc(res)
        sx, sy = _subpixel_parabola(res, mx, my)
        pos = np.array([x0 + sx + st2, y0 + sy + st2])  # centro del contenido hallado
        return pos, float(conf)

    # ------------------------------------------------------------------ #
    #  Relocalización tras salto (flyback)                                #
    # ------------------------------------------------------------------ #
    def _ajustar_a_nodo(self, S_medido, objetivo):
        """Suma a S_medido el múltiplo entero de vectores de red más cercano a objetivo."""
        km = np.round(self.base_inv @ (objetivo - S_medido))
        return S_medido + self.base @ km

    def _detectar_red_validada(self, gris):
        """Detección de discos APTA para relocalizar: exige respuesta comparable
        a la de discos nítidos (rechaza frames borroneados por el flyback, donde
        solo respondería la suciedad fija a la cámara) y que el conjunto forme
        una red periódica coherente (la suciedad no la forma).

        Devuelve (detecciones, fase_obs) o (None, None) si el frame no sirve.
        """
        detectados, resp = self._detectar_discos(gris, n_max=40)
        if len(detectados) < 12:
            return None, None
        med = float(np.median(resp))
        gate = 0.55 * self.resp_ref
        buenos = resp >= gate
        if med < gate or buenos.sum() < 12:
            return None, None
        detectados = detectados[buenos]

        # validación de estructura: fases (mod 1) en coordenadas de red
        U = detectados @ self.base_inv.T
        fase_obs = np.empty(2)
        for eje in (0, 1):
            fase_obs[eje] = _mediana_circular(U[:, eje], 1.0)
            residuos = np.abs(_wrap(U[:, eje] - fase_obs[eje], 1.0))
            if np.median(residuos) > 0.08:
                return None, None  # el conjunto no es una red -> frame inservible
        self.resp_ref = 0.5 * self.resp_ref + 0.5 * med  # adaptación a la iluminación
        return detectados, fase_obs

    def _completar_ensamble(self, gris, trackers, S_actual, detecciones=None):
        """Rellena el ensamble hasta n_trackers con discos frescos no ocupados."""
        if len(trackers) >= self.n_trackers:
            return trackers
        if detecciones is None:
            detecciones, _ = self._detectar_red_validada(gris)
            if detecciones is None:
                return trackers
        umbrales = np.linspace(0.45, 0.85, max(2, self.n_trackers)) * self.radio_busqueda
        for pos in detecciones:
            if len(trackers) >= self.n_trackers:
                break
            if all(np.hypot(*(pos - t["pos"])) > 0.7 * self.periodo_med for t in trackers):
                nt = self._capturar_template(gris, pos, S_actual,
                                             umbrales[len(trackers) % len(umbrales)])
                if nt is not None:
                    trackers.append(nt)
        return trackers

    def _fase_referencia(self, trackers):
        """Fase (mod 1, coords de red) de los nodos del ensamble llevados a S=0."""
        L = np.array([t["anclaje"] - t["S_cap"] for t in trackers])
        U_ref = L @ self.base_inv.T
        return np.array([_mediana_circular(U_ref[:, eje], 1.0) for eje in (0, 1)])

    def _resembrar_por_mediciones(self, gris, S_nuevo, mediciones, inliers):
        """Re-templatea el ensamble en las posiciones medidas (la apariencia de
        los discos cambió con la posición) tras un relock por medición NCC."""
        nuevos = []
        for (t, pos, conf), ok in zip(mediciones, inliers):
            if ok:
                nt = self._capturar_template(gris, pos, S_nuevo, t["umbral_ret"])
                if nt is not None:
                    nuevos.append(nt)
        if len(nuevos) < 5:
            return None
        return self._completar_ensamble(gris, nuevos, S_nuevo)

    def _relock_por_deteccion(self, gris, objetivo, trackers, fase_ref=None):
        """Relocalización de respaldo (ensamble perdido): redetecta la red y ancla
        su fase, en coordenadas de red, al nodo más cercano a la predicción.

        fase_ref debe venir del MISMO operador de detección (muestreada por el
        watchdog en frames sanos): así el sesgo entre el centro-de-blob de la
        detección y el centro-de-template del ensamble se cancela exactamente.
        Devuelve (S_nuevo, trackers_nuevos, fase_obs).
        """
        if not trackers:
            return None, None, None
        detectados, fase_obs = self._detectar_red_validada(gris)
        if detectados is None:
            return None, None, None

        if fase_ref is None:
            fase_ref = self._fase_referencia(trackers)
        obj_u = self.base_inv @ objetivo
        # nodos observados = fase_ref + S_u (mod 1)  =>  S_u = fase_obs - fase_ref (mod 1)
        u_nuevo = obj_u + _wrap(fase_obs - fase_ref - obj_u, 1.0)
        S_nuevo = self.base @ u_nuevo

        nuevos = self._sembrar_trackers(gris, S_nuevo, detectados)
        if len(nuevos) < 5:
            return None, None, None
        return S_nuevo, nuevos, fase_obs

    def _verificar_fila(self, gris, S_actual, ref_gris, ref_S, paso_esperado):
        """Verificación óptica del nodo tras un relock, contra la fila anterior.

        Compara frames tomados a la MISMA fase del scan (relock + N frames en
        ambas filas): el desplazamiento verdadero entre ellos es el paso de
        fila (~px, << medio período), así que el pico NCC más cercano al paso
        esperado identifica el nodo sin ambigüedad. Si la hipótesis del tracker
        difiere en ~un vector de red (aterrizaje en el borde de medio período,
        indecidible en el momento del relock), devuelve la corrección.
        """
        h, w = self.alto, self.ancho
        cy, cx = h // 2, w // 2
        hp = lambda g: g.astype(np.float32) - cv.GaussianBlur(
            g.astype(np.float32), (0, 0), 15)
        rad = int(0.75 * self.periodo_med)
        t = hp(ref_gris)[cy - 200:cy + 200, cx - 330:cx + 330]
        ex, ey = int(round(paso_esperado[0])), int(round(paso_esperado[1]))
        y0, y1 = cy - 200 + ey - rad, cy + 200 + ey + rad
        x0, x1 = cx - 330 + ex - rad, cx + 330 + ex + rad
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            return None
        s = hp(gris)[y0:y1, x0:x1]
        res = cv.matchTemplate(s, t, cv.TM_CCOEFF_NORMED)
        mx = cv.dilate(res, np.ones((25, 25), np.uint8))
        ys, xs = np.nonzero((res >= mx) & (res > 0.25))
        if not len(ys):
            return None
        # excluir el pico del patrón fijo (polvo), que está en desplazamiento 0
        lejos_del_polvo = (xs - (rad - ex)) ** 2 + (ys - (rad - ey)) ** 2 > 100
        xs, ys = xs[lejos_del_polvo], ys[lejos_del_polvo]
        if not len(ys):
            return None
        # pico más cercano al paso esperado = desplazamiento verdadero
        j = np.argmin((xs - rad) ** 2 + (ys - rad) ** 2)
        sx, sy = _subpixel_parabola(res, int(xs[j]), int(ys[j]))
        verdadero = np.array([ex + sx - rad, ey + sy - rad])
        # el paso real entre filas repite al comando del piezo: si el pico más
        # cercano quedó lejos de lo esperado, no hay evidencia utilizable
        if np.hypot(*(verdadero - paso_esperado)) > 25:
            return None
        hipotesis = S_actual - ref_S
        # corrección en nodos enteros de la red (0 si el relock estuvo bien)
        return self.base @ np.round(self.base_inv @ (verdadero - hipotesis))

    # ------------------------------------------------------------------ #
    #  Bucle principal                                                   #
    # ------------------------------------------------------------------ #
    def track(self):
        """Procesa el video completo. Devuelve dict con arrays por frame y metadatos."""
        cap = cv.VideoCapture(self.video_path)
        if self.frame_inicio > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_inicio)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("No se pudo leer el primer frame.")
        gris = self._gris(frame)

        # --- auto-calibración de escalas a partir de la red (rotación incluida) ---
        base = self._estimar_red_autocorr(gris)
        if base is not None:
            self.periodo_x = float(np.linalg.norm(base[:, 0]))
            self.periodo_y = float(np.linalg.norm(base[:, 1]))
        else:
            self.advertencias.append(
                "Autocorrelación 2D sin picos claros; uso perfiles por eje "
                "(la red debe estar aproximadamente alineada con la cámara).")
            self.periodo_x, self.periodo_y = self._estimar_periodo(gris)
            base = np.diag([self.periodo_x, self.periodo_y])
        self.periodo_med = 0.5 * (self.periodo_x + self.periodo_y)
        P = self.periodo_med
        self.sigma_blob = max(4.0, P / 10.0)                # escala del disco
        self.semitemplate = max(10, int(0.24 * P))          # template: lado 2*st
        self.radio_busqueda = max(12, int(0.29 * P))        # < medio período
        self.margen_seguro = self.semitemplate + self.radio_busqueda + 6
        umbral_salto_abs = max(22.0, 0.21 * P)              # paso legítimo máx ~16 px
        umbral_conf = 0.5

        discos_iniciales, resp_ini = self._detectar_discos(
            gris, n_max=150, min_dist=0.65 * P, margen=40)
        if len(resp_ini) < 5:
            cap.release()
            raise RuntimeError("No se detectaron discos en el primer frame del video.")
        # referencia de "cuánto responde un disco real y nítido" (se adapta luego)
        self.resp_ref = float(np.median(resp_ini))
        self.base = self._refinar_base(discos_iniciales, base)
        self.base_inv = np.linalg.inv(self.base)
        if self.verbose:
            a1, a2 = self.base[:, 0], self.base[:, 1]
            print(f"[i] Red: a1=({a1[0]:.1f},{a1[1]:.1f}) a2=({a2[0]:.1f},{a2[1]:.1f}) px | "
                  f"template {2*self.semitemplate} px | búsqueda ±{self.radio_busqueda} px")

        # --- estado global ---
        S = np.zeros(2)          # desplazamiento acumulado de la muestra
        V = np.zeros(2)          # velocidad estimada (EMA)
        trackers = self._sembrar_trackers(
            gris, S, discos_iniciales[np.all(np.abs(discos_iniciales - [self.ancho / 2, self.alto / 2])
                                             < [self.ancho / 2 - self.margen_seguro,
                                                self.alto / 2 - self.margen_seguro], axis=1)]
            if len(discos_iniciales) else None)
        if len(trackers) < 5:
            cap.release()
            raise RuntimeError(
                f"Detección inicial insuficiente ({len(trackers)} discos). "
                "Revisar foco/iluminación del video.")
        # punto de referencia reportado: disco más cercano al centro del frame
        centro_img = np.array([self.ancho / 2, self.alto / 2])
        P0 = min((t["anclaje"] for t in trackers),
                 key=lambda a: np.hypot(*(a - centro_img))).copy()
        if self.verbose:
            print(f"[i] {len(trackers)} discos en el ensamble. "
                  f"Referencia P0 = ({P0[0]:.0f}, {P0[1]:.0f}) px")

        # aprendizaje de la geometría del barrido
        inicio_fila = S.copy()          # arranque de los datos (objetivo del 1er relock)
        buffer_inicio = []              # muestras para estimar inicio_fila
        pasos_fila = []                 # vector de avance entre aterrizajes (aprendido)
        aterrizajes = []                # historia de aterrizajes de relocks
        frame_ultimo_relock = -10**9
        tend = {0: [], 1: []}           # signos de pasos recientes por eje
        dS_prev_mag = 0.0               # |dS| del frame anterior aceptado
        cooldown_relock = 0

        def objetivo_relock():
            """Aterrizaje predicho del flyback: el anterior + el paso de fila.

            Funciona igual para barridos de filas en x (flyback en x, paso en y)
            que para columnas en y (flyback en y, paso en x): el error de la
            predicción solo tiene que ser menor a medio período por eje, la fase
            de la red pone el subpíxel. NUNCA usar la medición envuelta del salto
            como objetivo: en el eje del flyback está aliased por construcción.
            Si el último aterrizaje desentona con la secuencia (commit temprano
            en una pausa del asentamiento), se predice desde el anterior.
            """
            if not aterrizajes:
                return inicio_fila.copy()
            paso = np.median(pasos_fila, axis=0) if pasos_fila else np.zeros(2)
            if len(aterrizajes) >= 3:
                # mediana de los últimos 3: un aterrizaje corrido de nodo o
                # medido en una etapa rara del asentamiento no arrastra la
                # predicción del siguiente (la mediana equivale a la fila k-1)
                return np.median(aterrizajes[-3:], axis=0) + 2.0 * paso
            return aterrizajes[-1] + paso
        pend_det = None                 # pendiente relock por detección
        pend_lite = None                # pendiente relock por medición
        relock_forzado = False          # perdido establecido: solo sale por relock
        frames_perdido = 0
        ref_fila = []                   # [(gris, S, n_fila)] a fase fija del scan
        verif_pendiente = 45            # frame en que verificar/tomar referencia
        n_fila = 0                      # contador de filas (relocks únicos)
        # fase de referencia de la red medida con el operador de DETECCIÓN,
        # normalizada a S=0 (se refresca en frames sanos vía watchdog)
        U0 = discos_iniciales @ self.base_inv.T
        fase_det_ref = np.array([_mediana_circular(U0[:, eje], 1.0) for eje in (0, 1)])

        n = self.total_frames - self.frame_inicio
        cols = {k: np.empty(n) for k in ("X", "Y", "conf")}
        cols["n_conf"] = np.empty(n, dtype=np.int32)
        estados = np.empty(n, dtype=object)
        t0 = time.perf_counter()
        i = 0
        abortado = False

        def registrar_relock(motivo, S_prev, S_nuevo, nuevos_trackers):
            nonlocal S, V, trackers, estado, tend, cooldown_relock, dS_prev_mag, \
                frame_ultimo_relock, relock_forzado, frames_perdido, pend_det, \
                pend_lite, verif_pendiente, n_fila
            relock_forzado = False
            frames_perdido = 0
            pend_det = pend_lite = None
            if i - frame_ultimo_relock > 60:
                n_fila += 1
            verif_pendiente = i + 45
            frame_abs = self.frame_inicio + i
            self.relocalizaciones.append((frame_abs, motivo, S_prev.copy(), S_nuevo.copy()))
            if self.verbose:
                print(f"[i] Frame {frame_abs}: relocalizado ({motivo}) "
                      f"S=({S_prev[0]:+.1f},{S_prev[1]:+.1f}) -> ({S_nuevo[0]:+.1f},{S_nuevo[1]:+.1f})")
            S, V = S_nuevo, np.zeros(2)
            trackers = nuevos_trackers
            estado = "relocalizado"
            # aprender el paso entre filas; un relock a < 30 frames del anterior es
            # una corrección en pleno vuelo/asentamiento, no un cambio de fila
            if aterrizajes and i - frame_ultimo_relock > 30:
                pasos_fila.append(S_nuevo - aterrizajes[-1])
            aterrizajes.append(S_nuevo.copy())
            frame_ultimo_relock = i
            tend = {0: [], 1: []}
            dS_prev_mag = 99.0  # el asentamiento arranca en movimiento
            # Durante el asentamiento no puede haber otro flyback (el dwell dura
            # segundos): los pasos grandes que siguen son movimiento real y deben
            # aceptarse, no disparar nuevas relocalizaciones.
            cooldown_relock = 30

        while True:
            if i > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                gris = self._gris(frame)

            cooldown_relock = max(0, cooldown_relock - 1)
            S_pred = S + V

            # --- medición del ensamble ---
            mediciones, S_estimados, respawnear = [], [], []
            for t in trackers:
                m = self._medir_tracker(gris, t, S_pred)
                if m is None:
                    respawnear.append(t)
                    continue
                pos, conf = m
                t["pos"], t["conf"] = pos, conf
                if conf >= umbral_conf:
                    mediciones.append((t, pos, conf))
                    S_estimados.append(t["S_cap"] + (pos - t["anclaje"]))

            estado = "perdido"
            conf_frame, n_inliers = 0.0, 0
            if len(S_estimados) >= 4:
                S_est = np.array(S_estimados)
                S_med = np.median(S_est, axis=0)
                inliers = np.hypot(*(S_est - S_med).T) < 3.0
                n_inliers = int(inliers.sum())
                S_med = np.median(S_est[inliers], axis=0) if inliers.any() else S
                conf_frame = float(np.median(
                    [c for (t, p, c), ok in zip(mediciones, inliers) if ok])) \
                    if inliers.any() else 0.0
                if n_inliers >= max(4, int(0.25 * len(trackers))) and conf_frame >= 0.55:
                    dS = S_med - S

                    # --- guardia contra saltos disfrazados (aliasing de la red) ---
                    # Un flyback envuelto módulo el período puede parecer un paso
                    # normal, pero siempre va CONTRA la tendencia de la fila (y los
                    # pasos legítimos medidos son <= 16 px, bajo el umbral absoluto).
                    # Solo cuentan los pasos "desde un dwell" (frame previo casi
                    # estático): la deriva de asentamiento post-flyback se mueve
                    # todos los frames y no debe armar ni disparar esta regla.
                    desde_dwell = dS_prev_mag < 2.0
                    sospechoso = np.hypot(*dS) > umbral_salto_abs
                    for eje in (0, 1):
                        t_ = tend[eje]
                        if len(t_) == 3 and len(set(t_)) == 1 and desde_dwell and \
                           abs(dS[eje]) > 6 and np.sign(dS[eje]) == -t_[0]:
                            sospechoso = True  # paso contra la tendencia de la fila

                    if (sospechoso and cooldown_relock == 0) or relock_forzado:
                        # Nunca se acepta un salto sospechoso, y desde que dispara
                        # queda bloqueada TODA aceptación hasta que un relock
                        # confirme (relock_forzado). Sin este bloqueo, un retorno
                        # en rampa lenta (p.ej. -26 px/frame en las calibraciones
                        # de y) hace deslizar el tracker nodo a nodo: tras ~4
                        # frames congelado, el patrón se corrió un período casi
                        # justo y el dS aparente (~1 px) se aceptaría — un
                        # deslizamiento de nodos enteros que ni el watchdog de
                        # fase puede ver. Congelado se espera el fin de la rampa
                        # y el nodo lo decide la predicción del aterrizaje.
                        # Se propone un candidato ajustado a nodo y se exige
                        # confirmación en 4 frames consecutivos casi estáticos
                        # (la MEDICIÓN cruda estática, no solo el candidato: en
                        # vuelo los candidatos consecutivos difieren y no
                        # confirman). Se exige además un match nítido (conf
                        # alta): en un frame borroneado la "medición" es basura.
                        relock_forzado = True
                        if conf_frame >= 0.7:
                            # La confirmación compara la MEDICIÓN CRUDA (S_med),
                            # no el candidato ajustado: si el asentamiento avanza
                            # ~período/4 por frame, el ajuste a nodo produce
                            # candidatos constantes con la muestra aún en
                            # movimiento (aliasing de la confirmación).
                            S_cand = self._ajustar_a_nodo(S_med, objetivo_relock())
                            if pend_lite is not None \
                               and i - pend_lite[0] == 1 \
                               and np.hypot(*(S_med - pend_lite[3])) < 3.0 \
                               and np.hypot(*(S_cand - pend_lite[1])) < 3.0:
                                if pend_lite[2] + 1 >= 3:
                                    # Validación anti-polvo: durante una rampa
                                    # borroneada el ensamble puede matchear el
                                    # patrón fijo con conf alta y S_med "estático"
                                    # (= posición congelada). Se exige que la RED
                                    # detectada exista (frame nítido) y que su
                                    # fase coincida con el candidato.
                                    det_v, fase_v = self._detectar_red_validada(gris)
                                    coincide = det_v is not None and np.abs(_wrap(
                                        fase_v - fase_det_ref - self.base_inv @ S_cand,
                                        1.0)).max() < 0.06
                                    nuevos = self._resembrar_por_mediciones(
                                        gris, S_cand, mediciones, inliers) \
                                        if coincide else None
                                    if nuevos is not None:
                                        S_prev = S.copy()
                                        registrar_relock("salto", S_prev, S_cand, nuevos)
                                        pend_lite = None
                                    else:
                                        pend_lite = (i, S_cand, 0, S_med)
                                else:
                                    pend_lite = (i, S_cand,
                                                 pend_lite[2] + 1, S_med)
                            else:
                                pend_lite = (i, S_cand, 0, S_med)
                        # si no se pudo, cae al relock por detección más abajo
                    else:
                        # --- aceptar la medición ---
                        V = 0.5 * V + 0.5 * dS
                        S = S_med
                        estado = "ok"
                        for eje in (0, 1):
                            if abs(dS[eje]) > 5 and desde_dwell:
                                tend[eje].append(int(np.sign(dS[eje])))
                                tend[eje] = tend[eje][-3:]
                        dS_prev_mag = float(np.hypot(*dS))

                        # estimación del arranque de los datos (objetivo del 1er relock)
                        if len(buffer_inicio) < 20:
                            buffer_inicio.append(S.copy())
                            if len(buffer_inicio) == 20:
                                inicio_fila = np.median(buffer_inicio, axis=0)

                        # mantenimiento: re-templateo escalonado. SOLO se
                        # re-templatea en el lugar a un tracker sano (conf alta):
                        # si su disco desapareció (p.ej. salió del borde de la
                        # zona con muestra), re-templatear capturaría fondo/polvo
                        # fijo y el tracker quedaría clavado como "disco
                        # fantasma" — se elimina y el watchdog repone después.
                        muertos = []
                        for t in trackers:
                            drift = np.hypot(*(S - t["S_cap"]))
                            t["malos"] = t["malos"] + 1 if t["conf"] < 0.45 else 0
                            if t["malos"] >= 3:
                                muertos.append(t)
                            elif drift > t["umbral_ret"] and t["conf"] >= 0.6:
                                nuevo = self._capturar_template(
                                    gris, t["pos"], S, t["umbral_ret"])
                                if nuevo is not None:
                                    t.update(nuevo)
                                    t["malos"] = 0
                        if muertos:
                            trackers = [t for t in trackers
                                        if all(t is not m for m in muertos)]

            # --- reposición reactiva: si el ensamble quedó corto (p.ej. el borde
            #     de la zona con discos barre el campo al final de la fila y los
            #     trackers mueren en tandas), se repone rápido con discos reales ---
            if estado == "ok" and len(trackers) < 0.7 * self.n_trackers and i % 10 == 4:
                det_t, _ = self._detectar_red_validada(gris)
                if det_t is not None:
                    trackers = self._completar_ensamble(gris, trackers, S, det_t)

            # --- watchdog anti-desincronización: la fase de la red detectada debe
            #     seguir a S. Atrapa tanto el anclaje al patrón fijo (polvo) como
            #     los flybacks aliased que la guardia no vio (envueltos en un paso
            #     pro-tendencia). De paso refresca la fase de referencia del
            #     operador de detección y repone el ensamble con discos reales.
            #     Cada 30 frames: ~0.3 ms/frame amortizado. Se evita en frames de
            #     paso rápido (la fase detectada a mitad del movimiento difiere de
            #     S por el tamaño del paso -> falsos positivos), pero se admite el
            #     movimiento lento de los barridos continuos, con umbral que
            #     escala con el movimiento actual. ---
            if estado == "ok" and dS_prev_mag < 6.0 and i % 30 == 29:
                det_w, fase_w = self._detectar_red_validada(gris)
                if det_w is not None:
                    fase_esp = fase_det_ref + self.base_inv @ S
                    umbral_wd = 0.12 + dS_prev_mag / self.periodo_med
                    if np.abs(_wrap(fase_w - fase_esp, 1.0)).max() > umbral_wd:
                        estado = "perdido"  # fuerza relocalización abajo
                        relock_forzado = True
                        self.advertencias.append(
                            f"Watchdog: ensamble desincronizado de la red en el frame "
                            f"{self.frame_inicio + i}; se relocalizó.")
                    else:
                        fase_det_ref = (fase_w - self.base_inv @ S) % 1.0
                        if len(trackers) < self.n_trackers:
                            trackers = self._completar_ensamble(gris, trackers, S, det_w)

            # pérdida sostenida -> el ensamble ya no es confiable: solo relock
            frames_perdido = frames_perdido + 1 if estado == "perdido" else 0
            if frames_perdido >= 5:
                relock_forzado = True

            # --- verificación óptica del nodo tras el relock ---
            # Si el aterrizaje cayó cerca del borde de medio período (el
            # asentamiento puede dejar la posición a ~P/2 de la predicción),
            # el nodo elegido pudo ser el equivocado. A fase fija del scan
            # (relock + 45) se contrasta contra la fila anterior: el paso
            # verdadero entre filas es chico y decide el nodo sin ambigüedad.
            if verif_pendiente is not None and i >= verif_pendiente and estado == "ok":
                if ref_fila:
                    # elegir la referencia cuyo paso esperado quede lejos del
                    # pico de polvo (>18 px) pero lejos del medio período (<45):
                    # con paso ~9 px eso significa comparar 2-4 filas atrás.
                    # El paso mediano se calcula sobre entradas sanas (|p|<60):
                    # los pares alrededor de un nodo corrido no deben polutarlo.
                    sanos = [p for p in pasos_fila if np.hypot(*p) < 60]
                    paso1 = np.median(sanos, axis=0) if sanos else np.zeros(2)
                    mejor = None
                    for g_ref, S_ref, fila_ref in reversed(ref_fila):
                        paso = paso1 * max(1, n_fila - fila_ref)
                        mag = np.hypot(*paso)
                        if 18 <= mag <= 45:
                            pena = abs(mag - 30)
                            if mejor is None or pena < mejor[0]:
                                mejor = (pena, g_ref, S_ref, paso)
                    corr = None
                    if mejor is not None:
                        _, g_ref, S_ref, paso = mejor
                        corr = self._verificar_fila(gris, S, g_ref, S_ref, paso)
                    if corr is not None and np.hypot(*corr) > 40:
                        S = S + corr
                        for t in trackers:
                            t["S_cap"] = t["S_cap"] + corr
                        # corregir retroactivamente los frames de esta fila
                        cols["X"][frame_ultimo_relock:i] += corr[0]
                        cols["Y"][frame_ultimo_relock:i] += corr[1]
                        if aterrizajes:
                            aterrizajes[-1] = aterrizajes[-1] + corr
                        if pasos_fila:
                            pasos_fila[-1] = pasos_fila[-1] + corr
                        frame_abs = self.frame_inicio + i
                        self.relocalizaciones.append(
                            (frame_abs, "verificacion", (S - corr).copy(), S.copy()))
                        if self.verbose:
                            print(f"[i] Frame {frame_abs}: nodo corregido por "
                                  f"verificación óptica ({corr[0]:+.0f},{corr[1]:+.0f}) px")
                ref_fila.append((gris.copy(), S.copy(), n_fila))
                ref_fila = ref_fila[-8:]
                verif_pendiente = None

            # --- pérdida total: relocalizar redetectando la red ---
            # Confirmación por CONSISTENCIA DE TRAYECTORIA en 3 frames: los
            # candidatos consecutivos deben diferir exactamente en el movimiento
            # medido por la fase de la red entre frames (válido porque el paso
            # entre frames es << medio período). Esto permite relocalizar aunque
            # el barrido siga en movimiento (cadencias rápidas o barrido
            # continuo); los frames borroneados del vuelo/asentamiento ya los
            # rechazan los gates de nitidez de la detección validada.
            if estado == "perdido" and cooldown_relock == 0:
                S_prev = S.copy()
                S_cand, nuevos, fase_cand = self._relock_por_deteccion(
                    gris, objetivo_relock(), trackers, fase_ref=fase_det_ref)
                if S_cand is not None:
                    consistente = False
                    if pend_det is not None and i - pend_det[0] == 1:
                        dS_det = self.base @ _wrap(fase_cand - pend_det[3], 1.0)
                        consistente = np.hypot(
                            *(S_cand - pend_det[1] - dS_det)) < 2.0
                    if consistente:
                        if pend_det[2] + 1 >= 2:
                            registrar_relock("perdida", S_prev, S_cand, nuevos)
                            conf_frame, n_inliers = 1.0, len(trackers)
                            pend_det = None
                        else:
                            pend_det = (i, S_cand, pend_det[2] + 1, fase_cand)
                    else:
                        pend_det = (i, S_cand, 0, fase_cand)
                # un frame sin candidato deja vencer el pendiente solo (i-1 != i)
            elif estado == "ok" and respawnear:
                # el disco salió del campo visual: se elimina y el watchdog
                # repone con discos reales detectados (solo donde hay muestra)
                trackers = [t for t in trackers
                            if all(t is not r for r in respawnear)]

            if self.debug_rango and self.debug_rango[0] <= i < self.debug_rango[1]:
                print(f"  dbg f{self.frame_inicio + i}: estado={estado:12s} "
                      f"trackers={len(trackers)} med={len(mediciones)} inl={n_inliers} "
                      f"conf={conf_frame:.2f} S=({S[0]:+.1f},{S[1]:+.1f}) "
                      f"V=({V[0]:+.2f},{V[1]:+.2f}) forz={relock_forzado}")

            cols["X"][i] = P0[0] + S[0]
            cols["Y"][i] = P0[1] + S[1]
            cols["conf"][i] = conf_frame
            cols["n_conf"][i] = n_inliers
            estados[i] = estado

            # --- preview opcional ---
            if self.mostrar_video and i % self.preview_cada == 0:
                disp = cv.resize(frame, (self.ancho // 2, self.alto // 2))
                for t in trackers:
                    x, y = (t["pos"] / 2).astype(int)
                    ok = t["conf"] >= umbral_conf
                    cv.rectangle(disp, (x - 12, y - 12), (x + 12, y + 12),
                                 (0, 255, 0) if ok else (255, 0, 255), 1)
                color = {"ok": (0, 255, 0), "perdido": (0, 0, 255),
                         "relocalizado": (0, 255, 255)}[estado]
                cv.putText(disp, f"frame {self.frame_inicio + i}  estado: {estado}  "
                                 f"X={cols['X'][i]:.1f} Y={cols['Y'][i]:.1f}",
                           (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv.imshow("Tracker de barridos v5 (q para abortar)", disp)
                if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                    abortado = True
                    self.advertencias.append(f"Trackeo abortado por el usuario en el frame {i}.")
                    i += 1
                    break

            i += 1
            if self.verbose and i % 1000 == 0:
                vel = i / (time.perf_counter() - t0)
                print(f"    frame {i}/{n}  ({vel:.0f} fps de procesamiento)")

        cap.release()
        if self.mostrar_video:
            cv.destroyAllWindows()
            for _ in range(4):
                cv.waitKey(1)

        dt = time.perf_counter() - t0
        frames_hechos = i
        if self.verbose:
            print(f"[i] {frames_hechos} frames en {dt:.1f} s "
                  f"({frames_hechos / max(dt, 1e-9):.0f} fps de procesamiento)")

        frames = self.frame_inicio + np.arange(frames_hechos)
        return {
            "X": cols["X"][:frames_hechos],
            "Y": cols["Y"][:frames_hechos],
            "Frame": frames,
            "Tiempo_seg": frames / self.fps,
            "confianza": cols["conf"][:frames_hechos],
            "n_confiables": cols["n_conf"][:frames_hechos],
            "estado": estados[:frames_hechos],
            "fps": self.fps,
            "periodo": (self.periodo_x, self.periodo_y),
            "base_red": self.base,
            "relocalizaciones": self.relocalizaciones,
            "advertencias": self.advertencias,
            "abortado": abortado,
            "fps_procesamiento": frames_hechos / max(dt, 1e-9),
        }
