#!/usr/bin/env python3
"""
Version améliorée (focus: détection des rangs) pour orthophotos de maïs.

Ce fichier fournit un moteur `RowDetector` réutilisable pour remplacer la logique
`detecter_rangs` de ton onglet Tkinter actuel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None


@dataclass
class RowDetectionParams:
    exg_morph: int = 3
    canny_sigma: float = 1.6
    hough_votes_ratio: float = 0.20
    hough_min_peaks: int = 24
    hough_max_peaks: int = 1200
    angle_tolerance_deg: float = 10.0
    cluster_eps_px: float = 10.0
    interrow_cm: float = 80.0
    interrow_tol_pct: float = 35.0


@dataclass
class RowDetectionResult:
    mask: np.ndarray
    lines: List[Tuple[float, float]]  # (theta_rad, rho)
    dominant_angle_deg: float
    interrow_px: Optional[float]
    debug: Dict[str, float]


class RowDetector:
    """Détection robuste de rangs basée ExG + Canny + Hough + clustering rho."""

    def __init__(self, params: Optional[RowDetectionParams] = None):
        self.p = params or RowDetectionParams()

    @staticmethod
    def _exg_mask(img_bgr: np.ndarray, morph_level: int) -> np.ndarray:
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)

        s = r + g + b + 1e-6
        exg = 2.0 * (g / s) - (r / s) - (b / s)

        exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu_thr, mask_otsu = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fusion Otsu + percentile pour éviter un seuil trop agressif sur images contrastées.
        p65 = np.percentile(exg_u8, 65)
        thr = int(round(0.5 * otsu_thr + 0.5 * p65))
        _, mask = cv2.threshold(exg_u8, thr, 255, cv2.THRESH_BINARY)

        k = max(1, int(morph_level))
        ks = 2 * k + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

        # Nettoyage des petits blobs (bruit).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            min_area = max(40, int(np.percentile(areas[1:], 25)))
            keep = np.zeros_like(mask)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[labels == i] = 255
            mask = keep

        return mask

    @staticmethod
    def _auto_canny(mask: np.ndarray, sigma: float) -> np.ndarray:
        blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.8, sigma), sigmaY=max(0.8, sigma))
        med = float(np.median(blur))
        lo = int(max(5, 0.66 * med))
        hi = int(min(255, 1.33 * med + 30))
        return cv2.Canny(blur, lo, hi, apertureSize=3, L2gradient=True)

    @staticmethod
    def _wrap_deg(d: float) -> float:
        while d < -90:
            d += 180
        while d > 90:
            d -= 180
        return d

    @staticmethod
    def _dominant_angle_deg(lines: np.ndarray) -> float:
        # cv2.HoughLines renvoie theta angle de la normale [0,pi].
        theta = lines[:, 0, 1]
        rho_abs = np.abs(lines[:, 0, 0])

        line_angles = np.degrees(theta) - 90.0
        line_angles = np.array([RowDetector._wrap_deg(a) for a in line_angles])

        bins = np.linspace(-90, 90, 181)
        hist, edges = np.histogram(line_angles, bins=bins, weights=rho_abs + 1.0)
        idx = int(np.argmax(hist))
        return float((edges[idx] + edges[idx + 1]) / 2.0)

    def _hough_with_fallback(self, edges: np.ndarray) -> np.ndarray:
        base = max(20, int(self.p.hough_votes_ratio * max(edges.shape)))
        thresholds = [base, int(base * 0.8), int(base * 0.6), int(base * 0.45)]

        best = None
        for t in thresholds:
            lines = cv2.HoughLines(edges, 1, np.pi / 180.0, max(8, t))
            if lines is None:
                continue
            if len(lines) >= self.p.hough_min_peaks:
                return lines[: self.p.hough_max_peaks]
            if best is None or len(lines) > len(best):
                best = lines

        if best is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return best[: self.p.hough_max_peaks]

    def _cluster_rho(
        self,
        selected: np.ndarray,
        gsd_cm: float,
    ) -> List[Tuple[float, float]]:
        if selected.size == 0:
            return []

        thetas = selected[:, 0, 1]
        rhos = selected[:, 0, 0]

        # DBSCAN sur rho 1D pour fusionner les lignes du même rang.
        eps = float(max(2.0, self.p.cluster_eps_px))
        if DBSCAN is not None:
            labels = DBSCAN(eps=eps, min_samples=1).fit_predict(rhos.reshape(-1, 1))
        else:
            # Fallback sans sklearn: clustering glouton 1D.
            order = np.argsort(rhos)
            labels = -np.ones(len(rhos), dtype=int)
            cur = 0
            last_rho = rhos[order[0]]
            labels[order[0]] = cur
            for idx in order[1:]:
                if abs(rhos[idx] - last_rho) > eps:
                    cur += 1
                labels[idx] = cur
                last_rho = rhos[idx]

        rows: List[Tuple[float, float]] = []
        for lab in sorted(set(labels.tolist())):
            idx = labels == lab
            theta_m = float(np.median(thetas[idx]))
            rho_m = float(np.median(rhos[idx]))
            rows.append((theta_m, rho_m))

        rows.sort(key=lambda x: x[1])

        # Contrôle de cohérence inter-rang (si GSD disponible).
        if gsd_cm > 0 and len(rows) >= 3:
            expected_px = self.p.interrow_cm / gsd_cm
            tol = self.p.interrow_tol_pct / 100.0
            d = np.diff([r[1] for r in rows])
            lo = expected_px * (1 - tol)
            hi = expected_px * (1 + tol)
            if lo > 1 and hi > lo:
                keep = [rows[0]]
                for row in rows[1:]:
                    delta = abs(row[1] - keep[-1][1])
                    if lo <= delta <= 1.8 * hi:
                        keep.append(row)
                    elif delta > 1.8 * hi:
                        # Saut trop grand: on garde aussi pour ne pas perdre un vrai rang isolé.
                        keep.append(row)
                rows = keep

        if len(rows) >= 2:
            # Fixer angle commun pour éviter croisements visuels sur l'overlay.
            theta_common = float(np.median([r[0] for r in rows]))
            rows = [(theta_common, r[1]) for r in rows]

        return rows

    def detect(
        self,
        img_bgr: np.ndarray,
        gsd_cm: float,
        forced_line_angle_deg: Optional[float] = None,
    ) -> RowDetectionResult:
        mask = self._exg_mask(img_bgr, self.p.exg_morph)
        edges = self._auto_canny(mask, self.p.canny_sigma)
        lines = self._hough_with_fallback(edges)

        if lines.size == 0:
            return RowDetectionResult(mask=mask, lines=[], dominant_angle_deg=0.0, interrow_px=None, debug={"hough": 0})

        dominant_line_angle = self._dominant_angle_deg(lines)
        target_line_angle = forced_line_angle_deg if forced_line_angle_deg is not None else dominant_line_angle

        # Filtre angulaire.
        tol = max(0.0, self.p.angle_tolerance_deg)
        normals = np.degrees(lines[:, 0, 1])
        line_angles = np.array([self._wrap_deg(a - 90.0) for a in normals])

        if tol == 0:
            selected = lines
        else:
            d = np.abs(line_angles - target_line_angle)
            d = np.minimum(d, 180 - d)
            selected = lines[d <= tol]
            if len(selected) < 6:
                # Tolérance trop stricte: fallback auto un peu plus large.
                d2 = np.abs(line_angles - dominant_line_angle)
                d2 = np.minimum(d2, 180 - d2)
                selected = lines[d2 <= max(tol, 14.0)]

        rows = self._cluster_rho(selected, gsd_cm)

        inter = None
        if len(rows) >= 2:
            inter = float(np.median(np.abs(np.diff([r[1] for r in rows]))))

        dbg = {
            "hough": float(len(lines)),
            "selected": float(len(selected)),
            "dominant_line_angle": float(dominant_line_angle),
            "target_line_angle": float(target_line_angle),
        }
        return RowDetectionResult(mask=mask, lines=rows, dominant_angle_deg=target_line_angle, interrow_px=inter, debug=dbg)


# --- Fonctions utilitaires pour brancher rapidement dans ton onglet Tkinter ---

def rang_vers_pts(theta_rad: float, rho: float, H: int, W: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Convertit (theta, rho) Hough en 2 points sur les bords de l'image."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)

    pts: List[Tuple[float, float]] = []
    if abs(s) > 1e-6:
        for x in [0.0, float(W - 1)]:
            y = (rho - x * c) / s
            if -1 <= y <= H:
                pts.append((x, y))
    if abs(c) > 1e-6:
        for y in [0.0, float(H - 1)]:
            x = (rho - y * s) / c
            if -1 <= x <= W:
                pts.append((x, y))

    if len(pts) < 2:
        return None

    # Dédupliquer et prendre les 2 plus éloignés.
    uniq: List[Tuple[float, float]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1 and abs(p[1] - q[1]) < 1 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None

    max_d = -1.0
    pair = (uniq[0], uniq[1])
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
            if d > max_d:
                max_d = d
                pair = (uniq[i], uniq[j])

    p1 = (int(round(pair[0][0])), int(round(pair[0][1])))
    p2 = (int(round(pair[1][0])), int(round(pair[1][1])))
    return p1, p2


if __name__ == "__main__":
    print(
        "Ce module est prêt à être branché dans ton UI Tkinter.\n"
        "Utilise RowDetector.detect(...) à la place de ton detecter_rangs actuel."
    )
