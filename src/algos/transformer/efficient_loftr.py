"""
Efficient LoFTR helpers
----------------------
Usage
-----
    # 1) initialise once (lazy singleton held inside the module)
    src_pts, dst_pts, conf = efficient_loftr.match_eloftr(img0, img1,
                                                          weights="/path/to/eloftr_outdoor.ckpt")

    # 2) optional: keep only matches spanning the ROI
    src_roi, dst_roi, conf_roi = efficient_loftr.filter_matches_to_roi(
        src_pts, dst_pts, conf, roi_polygon, img0.shape[:2]
    )

    # 3) optional AdaLAM, homography, … identical to your current pipeline
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Union, Optional

import cv2
import numpy as np
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch.serialization

# ════════════════════════════════════════════════════════════════════════════
# Efficient-LoFTR core ─ we use the upstream project as a Git sub-module
#   your_repo/src/_experiments/_ext/EfficientLoFTR/
# add that directory to PYTHONPATH once in your __init__.py or rootutils.setup()
# ════════════════════════════════════════════════════════════════════════════
from src._ext.EfficientLoFTR.src.loftr.loftr import LoFTR
from src._ext.EfficientLoFTR.src.config.default import get_cfg_defaults
from src._ext.EfficientLoFTR.src.utils.misc import lower_config

# --------------------------------------------------------------------------- #
#                                PARAMS                                       #
# --------------------------------------------------------------------------- #
_DEFAULT_MAX_EDGE = 1024          # resize so that max(H, W) ≤ this (keeps aspect)
_PAD_MULTIPLE     = 8             # EfficientLoFTR expects H, W ⩰ 0 (mod 8)

# --------------------------------------------------------------------------- #
#                     INFERENCE WRAPPER (singleton)                           #
# --------------------------------------------------------------------------- #
class _EfficientLoFTRWrapper:
    """One-time initialisation wrapper around Efficient LoFTR."""
    def __init__(
        self,
        weights: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        max_edge: int = _DEFAULT_MAX_EDGE,
        conf_thresh: float = 0.25,
        half: bool = False,
    ) -> None:
        self.device      = torch.device(device if device != "auto"
                                        else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_edge    = max_edge
        self.conf_thresh = conf_thresh

        # --- build config ----------------------------------------------------
        cfg = get_cfg_defaults()                 # yacs CfgNode (uppercase keys)
        cfg.LOFTR.HALF = half                   # allow FP16 if desired
        cfg.freeze()
        elo_cfg = lower_config(cfg.LOFTR)       # -> plain dict with *lower-case* keys

        # ensure RoPEPositionEncodingSine gets a numeric 'npe' list of length 4:
        # [train_H, train_W, test_H, test_W]
        max_coarse = self.max_edge // _PAD_MULTIPLE
        coarse_cfg = elo_cfg.get("coarse", {})
        if coarse_cfg.get("npe", None) is None:
            # use the same 128×128 grid for train & test
            coarse_cfg["npe"] = [max_coarse,
                                 max_coarse,
                                 max_coarse,
                                 max_coarse]
            elo_cfg["coarse"] = coarse_cfg

        # --- model -----------------------------------------------------------
        self.matcher = LoFTR(elo_cfg).to(self.device).eval()
        
        # Add safe globals for PyTorch Lightning checkpoints
        torch.serialization.add_safe_globals([ModelCheckpoint])
        try:
            # First try with weights_only
            ckpt = torch.load(str(weights), map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to regular loading if weights_only fails
            ckpt = torch.load(str(weights), map_location=self.device)
            
        state_dict = ckpt.get("state_dict", ckpt)    # works for both styles
        self.matcher.load_state_dict(state_dict, strict=False)

    # ---------------------------- helpers ---------------------------------- #
    @staticmethod
    def _pad_to_multiple(img: np.ndarray, m: int = _PAD_MULTIPLE) -> np.ndarray:
        h, w = img.shape[:2]
        pad_h, pad_w = (-h) % m, (-w) % m
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    def _resize_keep_aspect(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        scale = min(1.0, self.max_edge / max(h, w))
        new_h = int(round(h * scale / _PAD_MULTIPLE)) * _PAD_MULTIPLE
        new_w = int(round(w * scale / _PAD_MULTIPLE)) * _PAD_MULTIPLE
        if (new_h, new_w) != (h, w):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale

    @staticmethod
    def _to_tensor_gray(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return torch.from_numpy(gray)[None, None].to(device)   # (1,1,H,W)

    # ----------------------------- API ------------------------------------- #
    @torch.inference_mode()
    def match(
        self,
        img0_bgr: np.ndarray,
        img1_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            src_pts, dst_pts : (N,2)  float32 in original pixel coords  (x,y order)
            mconf            : (N,)   confidence scores ∈ [0,1]
        """
        img0_r, s0 = self._resize_keep_aspect(img0_bgr)
        img1_r, s1 = self._resize_keep_aspect(img1_bgr)

        img0_r = self._pad32(img0_r)
        img1_r = self._pad32(img1_r)

        batch = {
            "image0": self._to_tensor_gray(img0_r, self.device),
            "image1": self._to_tensor_gray(img1_r, self.device),
        }
        self.matcher(batch)

        mk0  = batch["mkpts0_f"].cpu().numpy()   # coarse→fine refined kpts
        mk1  = batch["mkpts1_f"].cpu().numpy()
        mconf = batch["mconf"].cpu().numpy()

        keep = mconf > self.conf_thresh
        if keep.sum() == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        # back-scale to original resolution
        return mk0[keep] / s0, mk1[keep] / s1, mconf[keep]


    # ensure divisible by 32 so that the 3×stride-2 downsamples line up
    def _pad32(self, im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        pad_h = (-h) % 32
        pad_w = (-w) % 32
        if pad_h or pad_w:
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        return im

# --------------------------------------------------------------------------- #
#                           PUBLIC CONVENIENCE API                           #
# --------------------------------------------------------------------------- #
_WRAPPER: Optional[_EfficientLoFTRWrapper] = None  # global singleton


def match_eloftr(
    img0: np.ndarray,
    img1: np.ndarray,
    weights: str = "/path/to/eloftr_outdoor.ckpt",
    conf_thresh: float = 0.25,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin convenience layer: returns src_pts, dst_pts, conf  (all np.ndarrays).
    """
    global _WRAPPER
    if _WRAPPER is None:
        _WRAPPER = _EfficientLoFTRWrapper(weights, device=device, conf_thresh=conf_thresh)
    return _WRAPPER.match(img0, img1)


# --------------------------------------------------------------------------- #
#        ROI filtering & RANSAC helpers (identical to your LoFTR ones)        #
# --------------------------------------------------------------------------- #
def filter_matches_to_roi(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    conf_pts: np.ndarray,
    roi_pts: np.ndarray,
    img_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only those correspondences whose **both endpoints** fall inside `roi_pts`.
    `roi_pts` is a polygon in (x,y) image coordinates.
    """
    if roi_pts is None:
        return src_pts, dst_pts, conf_pts

    mask_img = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask_img, [np.asarray(roi_pts, np.int32)], 1)

    idx0 = mask_img[np.round(src_pts[:,1]).astype(int), np.round(src_pts[:,0]).astype(int)].astype(bool)
    idx1 = mask_img[np.round(dst_pts[:,1]).astype(int), np.round(dst_pts[:,0]).astype(int)].astype(bool)
    keep = idx0 & idx1

    return src_pts[keep], dst_pts[keep], conf_pts[keep]
