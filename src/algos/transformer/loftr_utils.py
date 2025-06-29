import os

import numpy as np
import cv2
import matplotlib


# visualisation
def vis_matches_loftr(
    img1: np.ndarray,
    img2: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    conf: np.ndarray = None,
    roi_pts: np.ndarray = None,
    root: str = None,
    seq_num: str = "",
    idx: int = 0,
    show: bool = True,
    save: bool = True,
    window_title: str = "LoFTR Matches"
) -> np.ndarray:
    """
    Visualize LoFTR matches by concatenating img1 and img2 side by side,
    drawing the ROI polygon on each, then drawing lines between matched points.
    Matches are colored according to their confidence scores (if provided).

    Args:
        img1, img2:           BGR images.
        src_pts (Nx2 float):  points in img1 (x,y).
        dst_pts (Nx2 float):  points in img2 (x,y).
        conf (N,) float:      confidence scores for each match (0..1).
        roi_pts (Mx2 int):    polygon vertices in img1/img2 space.
        root (str):           base output dir; if None, no files are written.
        seq_num (str):        subfolder name under root.
        idx (int):            index of this match pair (for filename).
        show (bool):          call `cv2.imshow`.
        save (bool):          write jpeg to disk.
        window_title (str):   title for the cv2 window.

    Returns:
        The visualization as a single NumPy image (h x (w1+w2) x 3).
    """
    # draw ROI if given
    img1_vis = img1.copy()
    img2_vis = img2.copy()
    if roi_pts is not None:
        roi = np.array(roi_pts, dtype=np.int32)
        cv2.polylines(img1_vis, [roi], True, (0,255,0), 2)
        cv2.polylines(img2_vis, [roi], True, (0,255,0), 2)

    # pad both images to the same height
    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    H = max(h1, h2)
    def pad_to_height(img, target_h):
        h, w = img.shape[:2]
        if h < target_h:
            pad = target_h - h
            # black bottom padding
            return cv2.copyMakeBorder(img, 0, pad, 0, 0,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[0,0,0])
        return img
    img1_pad = pad_to_height(img1_vis, H)
    img2_pad = pad_to_height(img2_vis, H)

    # concatenate left/right
    canvas = np.hstack([img1_pad, img2_pad])

    # prepare colors per match
    if conf is not None and len(conf) == len(src_pts):
        # use matplotlib colormap
        cmap = matplotlib.cm.get_cmap('jet')
        # RGBA floats
        cols = cmap(conf)
        # convert to BGR ints
        cols_bgr = [(
            int(c[2]*255),
            int(c[1]*255),
            int(c[0]*255)
        ) for c in cols]
    else:
        # default yellow for all
        cols_bgr = [(0,255,255)] * len(src_pts)

    # draw matches
    for (x1,y1), (x2,y2), color in zip(src_pts, dst_pts, cols_bgr):
        p1 = (int(round(x1)),   int(round(y1)))
        p2 = (int(round(x2)) + w1, int(round(y2)))
        cv2.circle(canvas, p1, 4, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, color, 1, cv2.LINE_AA)
        cv2.line  (canvas, p1, p2, color, 1, cv2.LINE_AA)

    # save to disk if requested
    if save and root is not None:
        out_dir = os.path.join(root, seq_num, "loftr_matches")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"matches_img_{idx:03d}_{idx+1:03d}.jpg")
        cv2.imwrite(fname, canvas)

    # display if requested
    if show:
        cv2.imshow(window_title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas


def vis_blend_loftr(
    img1: np.ndarray,
    img2: np.ndarray,
    homography: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    conf: np.ndarray = None,
    roi_pts: np.ndarray = None,
    invert_direction: bool = False,
    alpha: float = 0.5,
    root: str = None,
    seq_num: str = "",
    idx: int = 0,
    show: bool = True,
    save: bool = True,
    window_title: str = "Blended Homography"
) -> np.ndarray:
    """
    Warp img(source) into img(destination) using homography, blend them,
    draw ROI on the source, and overlay src→dst matches colored by confidence.

    Args:
        img1, img2:      BGR images.
        homography:      3x3 homography mapping src→dst.
        src_pts (Nx2):   [x,y] in source image coords.
        dst_pts (Nx2):   [x,y] in destination image coords.
        conf (N,)        confidence scores ∈ [0,1] for each match.
        roi_pts (Mx2)    polygon in *source* image coords.
        invert_direction: if True, treat img2 as source and img1 as destination.
        alpha            blending weight for destination image.
        root, seq_num, idx, show, save, window_title: as usual.

    Returns:
        The blended BGR image with matches overlaid.
    """
    # decide which image is source / dest
    if invert_direction:
        src_img, dst_img = img2.copy(), img1.copy()
        src, dst = dst_pts, src_pts
    else:
        src_img, dst_img = img1.copy(), img2.copy()
        src, dst = src_pts, dst_pts

    # draw ROI on source
    if roi_pts is not None:
        roi = np.array(roi_pts, dtype=np.int32)
        cv2.polylines(src_img, [roi], True, (0,255,0), 2)

    # warp source into destination frame
    h_dst, w_dst = dst_img.shape[:2]
    warped = cv2.warpPerspective(src_img, homography, (w_dst, h_dst))

    # if grayscale, convert to BGR
    if warped.ndim==2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    if dst_img.ndim==2:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)

    # blend
    blended = cv2.addWeighted(dst_img, alpha, warped, 1-alpha, 0)

    # prepare colors for matches
    if conf is not None and len(conf)==len(src):
        cmap = matplotlib.cm.get_cmap('jet')
        colors = (cmap(conf)[:,:3]*255).astype(np.uint8)[:,::-1]  # RGB→BGR
    else:
        colors = np.tile(np.array([0,255,255],dtype=np.uint8), (len(src),1))

    # overlay matches
    w_offset = 0  # already in same frame
    for (x0,y0),(x1,y1),col in zip(src, dst, colors):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.circle(blended, p0, 3, tuple(map(int,col)), -1, cv2.LINE_AA)
        cv2.circle(blended, p1, 3, tuple(map(int,col)), -1, cv2.LINE_AA)
        cv2.line  (blended, p0, p1, tuple(map(int,col)), 1, cv2.LINE_AA)

    # save if desired
    if save and root is not None:
        out_dir = os.path.join(root, seq_num, "blend")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"blend_{idx:04d}.jpg")
        cv2.imwrite(fname, blended)

    # show if desired
    if show:
        cv2.imshow(window_title, blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return blended