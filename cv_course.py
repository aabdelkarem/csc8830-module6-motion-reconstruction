#!/usr/bin/env python3
"""CSc 8830 practical-assignment toolkit.
The blur command was executed and validated on 18 November 2026.
The classical thermal branch ran on a real public dog image on 19 November.
Thirteen synthetic integration checks passed; Aya's real captures remain pending.
Save the complete source as cv_course.py. Python >=3.10.
Install: python -m pip install numpy scipy matplotlib pillow opencv-python
Run: python cv_course.py --help
All real-data experiments require a JSON configuration. Paths are relative
to the working directory. Generated results must be inspected before use.
No default personal measurements, fabricated experimental data, or automatic
Classroom/GitHub uploads are included. The blur command has a clearly labelled
synthetic demonstration image because its task does not require a capture.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import textwrap
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
SOURCES = {
    "blur": "SciPy fftconvolve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html",
    "calibration": "OpenCV calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html",
    "flow": "OpenCV optical flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html",
    "sam2": "Meta SAM2: https://github.com/facebookresearch/sam2"
}
def cv():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python in this Python environment.") from exc
def outdir(config):
    p = Path(config["out"])
    p.mkdir(parents=True, exist_ok=True)
    return p
def read_image(path):
    c = cv()
    a = c.imread(str(path), c.IMREAD_COLOR)
    if a is None:
        raise ValueError("Cannot read image: " + str(path))
    return a
def as_points(value, n=None):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("Points must be finite [x, y] pairs.")
    if n is not None and len(a) != n:
        raise ValueError("Incorrect number of points.")
    return a
def check_points(p, width, height):
    if np.any(p[:, 0] < 0) or np.any(p[:, 0] >= width):
        raise ValueError("Point x is outside image.")
    if np.any(p[:, 1] < 0) or np.any(p[:, 1] >= height):
        raise ValueError("Point y is outside image.")
def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False),
                          encoding="utf-8")
def report(out, title, sections, pictures=()):
    """Write an execution report. It is not a complete submission by itself."""
    with PdfPages(out / "execution_report.pdf") as pdf:
        page = None
        y = 0
        def new_page():
            nonlocal page, y
            if page is not None:
                pdf.savefig(page)
                plt.close(page)
            page = plt.figure(figsize=(8.27, 11.69))
            page.text(.08, .95, title, fontsize=15, weight="bold")
            y = .90
        new_page()
        for heading, body in sections:
            if y < .16:
                new_page()
            page.text(.08, y, heading, fontsize=11, weight="bold")
            y -= .03
            for line in str(body).splitlines():
                for wrapped in textwrap.wrap(line, 95) or [""]:
                    if y < .08:
                        new_page()
                    page.text(.08, y, wrapped, fontsize=9)
                    y -= .019
            y -= .02
        pdf.savefig(page)
        plt.close(page)
        for path in pictures:
            f, ax = plt.subplots(figsize=(8.27, 8.27))
            ax.imshow(np.asarray(Image.open(path)))
            ax.set_title(Path(path).name)
            ax.axis("off")
            f.tight_layout()
            pdf.savefig(f)
            plt.close(f)
def calibration_load(path):
    z = np.load(path)
    K, dist, size = z["K"], z["dist"], tuple(map(int, z["image_size"]))
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera matrix.")
    return K, dist, size
def calibrate(cfg):
    """Checkerboard calibration; cols/rows are INNER corners."""
    c, out = cv(), outdir(cfg)
    cols, rows = int(cfg["cols"]), int(cfg["rows"])
    square = float(cfg["square_m"])
    if min(cols, rows) < 3 or square <= 0:
        raise ValueError("Invalid checkerboard geometry.")
    paths = sorted(Path(cfg["image_dir"]).glob(cfg.get("pattern", "*.jpg")))
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    objects, pixels, accepted, rejected = [], [], [], []
    size = None
    criteria = (c.TERM_CRITERIA_EPS + c.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for path in paths:
        a = read_image(path)
        gray = c.cvtColor(a, c.COLOR_BGR2GRAY)
        s = gray.shape[::-1]
        if size is None:
            size = s
        if s != size:
            raise ValueError("All calibration images must have the same resolution.")
        ok, corners = c.findChessboardCorners(
            gray, (cols, rows),
            c.CALIB_CB_ADAPTIVE_THRESH + c.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            rejected.append(str(path))
            continue
        corners = c.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objects.append(obj.copy())
        pixels.append(corners)
        accepted.append(str(path))
        c.drawChessboardCorners(a, (cols, rows), corners, ok)
        c.imwrite(str(out / ("corners_" + path.stem + ".png")), a)
    if len(accepted) < 10:
        raise ValueError("Need at least 10 successful varied calibration views; found "
                         + str(len(accepted)))
    rms, K, dist, rvecs, tvecs = c.calibrateCamera(
        objects, pixels, size, None, None)
    per_view = []
    for p, o, measured, r, t in zip(accepted, objects, pixels, rvecs, tvecs):
        projected, _ = c.projectPoints(o, r, t, K, dist)
        error = projected.reshape(-1, 2) - measured.reshape(-1, 2)
        per_view.append({"image": p,
                         "rms_px": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))})
    np.savez(out / "calibration.npz", K=K, dist=dist,
             image_size=np.array(size), rvecs=np.array(rvecs), tvecs=np.array(tvecs))
    result = {"rms_px": float(rms), "K": K.tolist(), "dist": dist.tolist(),
              "image_size": list(size), "accepted": accepted,
              "rejected": rejected, "per_view": per_view,
              "square_m": square, "inner_corners": [cols, rows],
              "opencv": c.__version__}
    save_json(out / "metrics.json", result)
    report(out, "Smartphone calibration", [
        ("Inputs", "Measured square size: " + str(square) + " m. Successful views: "
         + str(len(accepted))),
        ("Calibration fit", json.dumps(result, indent=2)),
        ("Validation limitation", "Reprojection residual is an in-sample fit diagnostic. "
         "Inspect corner overlays and validate dimensions on independently measured objects."),
        ("Reference", SOURCES["calibration"])])
def dimensions(cfg):
    """Parallel-plane dimensional measurement at independently measured axial Z."""
    c, out = cv(), outdir(cfg)
    a = read_image(cfg["image"])
    K, dist, size = calibration_load(cfg["calibration"])
    if a.shape[1::-1] != size:
        raise ValueError("Object and calibration image resolutions must match.")
    z = float(cfg["depth_m"])
    if z <= 2:
        raise ValueError("Assignment requires distance GREATER THAN 2 metres.")
    if not cfg.get("frontoparallel_confirmed", False):
        raise ValueError("Confirm the object plane is parallel to the image plane.")
    p = as_points(cfg["corners_tl_tr_br_bl"], 4)
    check_points(p, *size)
    q = c.undistortPoints(p.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1) * z
    width, height = float((sides[0] + sides[2]) / 2), float((sides[1] + sides[3]) / 2)
    result = {"width_m": width, "height_m": height, "depth_m": z,
              "four_side_lengths_m": sides.tolist(), "input_sha256": fingerprint(cfg["image"]),
              "assumption": "Planar, frontoparallel object at constant axial depth"}
    for name, estimate in [("width", width), ("height", height)]:
        truth = float(cfg["true_" + name + "_m"])
        if truth <= 0:
            raise ValueError("Ground-truth dimension must be positive.")
        result[name + "_ground_truth_m"] = truth
        result[name + "_signed_error_m"] = estimate - truth
        result[name + "_absolute_error_pct"] = 100 * abs(estimate - truth) / truth
    c.polylines(a, [np.rint(p).astype(np.int32)], True, (0, 255, 0), 3)
    c.putText(a, "W=%.3fm H=%.3fm Z=%.3fm" % (width, height, z),
              (15, 40), c.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
    c.imwrite(str(out / "measurement.png"), a)
    save_json(out / "metrics.json", result)
    report(out, "Module 2: perspective measurement", [
        ("Method", "Undistort corners to normalized coordinates; each planar length "
         "equals depth times its normalized image span. Average opposite sides."),
        ("Measured results", json.dumps(result, indent=2)),
        ("Limitations", "Depth means axial distance from camera centre. Object tilt, "
         "autofocus changes, inaccurate calibration and click errors can bias dimensions."),
        ("Reference", SOURCES["calibration"])], [out / "measurement.png"])
def fft_full(image, kernel):
    shape = tuple(np.array(image.shape) + np.array(kernel.shape) - 1)
    product = np.fft.fft2(image, s=shape) * np.fft.fft2(kernel, s=shape)
    return np.fft.ifft2(product).real
def blur(cfg):
    """Independent spatial convolution and explicitly zero-padded FFT product."""
    from scipy.signal import convolve2d
    out = outdir(cfg)
    if cfg.get("image"):
        a = np.asarray(Image.open(cfg["image"]).convert("L"), dtype=np.float64) / 255
        provenance = "User-supplied image; SHA256=" + fingerprint(cfg["image"])
    else:
        yy, xx = np.mgrid[0:192, 0:256]
        a = .15 + .6 * (((xx // 16 + yy // 16) % 2).astype(float))
        a[(xx - 170) ** 2 + (yy - 85) ** 2 < 36 ** 2] = 1
        a[20:60, 25:100] = .3
        provenance = "Deterministic synthetic test pattern, not a camera capture"
    k, sigma = int(cfg.get("kernel_size", 15)), float(cfg.get("sigma", 2.5))
    if k < 3 or k % 2 != 1 or sigma <= 0:
        raise ValueError("Use an odd kernel >=3 and positive sigma.")
    v = np.arange(k) - k // 2
    K = np.exp(-(v[:, None] ** 2 + v[None, :] ** 2) / (2 * sigma ** 2))
    K /= K.sum()
    start = time.perf_counter()
    spatial = convolve2d(a, K, mode="full", boundary="fill", fillvalue=0)
    t_spatial = time.perf_counter() - start
    start = time.perf_counter()
    frequency = fft_full(a, K)
    t_fft = time.perf_counter() - start
    difference = spatial - frequency
    np.testing.assert_allclose(spatial, frequency, rtol=1e-10, atol=1e-11)
    # An asymmetric kernel exposes convolution/correlation reversal mistakes.
    rng = np.random.default_rng(8830)
    random_image = rng.normal(size=(9, 11))
    asym = np.array([[1., 2., -1.], [0., 4., 3.]])
    np.testing.assert_allclose(fft_full(random_image, asym),
                               convolve2d(random_image, asym, mode="full"),
                               rtol=1e-10, atol=1e-11)
    impulse = np.zeros((7, 8))
    impulse[0, 0] = 1
    imp = fft_full(impulse, asym)
    np.testing.assert_allclose(imp[:2, :3], asym, atol=1e-11)
    np.testing.assert_allclose(spatial.sum(), a.sum() * K.sum(), rtol=1e-10)
    metrics = {
        "input": provenance, "input_shape": list(a.shape), "kernel_size": k,
        "sigma": sigma, "full_shape": list(spatial.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "spatial_seconds_single_run": t_spatial, "fft_seconds_single_run": t_fft,
        "validation": "full output, asymmetric kernel, impulse origin and sum checks passed"}
    pad = k // 2
    crop = (slice(pad, pad + a.shape[0]), slice(pad, pad + a.shape[1]))
    f, axes = plt.subplots(2, 3, figsize=(12, 7))
    images = [a, spatial[crop], frequency[crop], np.abs(difference[crop]),
              np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a)))),
              np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(K, s=a.shape))))]
    labels = ["Input", "Spatial convolution", "Fourier equivalent",
              "Absolute difference (autoscaled)", "Image spectrum", "Kernel spectrum"]
    for ax, im, label in zip(axes.flat, images, labels):
        ax.imshow(im, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    f.tight_layout()
    f.savefig(out / "comparison.png", dpi=160)
    plt.close(f)
    np.savez(out / "arrays.npz", input=a, kernel=K, spatial=spatial, fourier=frequency)
    save_json(out / "metrics.json", metrics)
    report(out, "Module 3: convolution theorem", [
        ("Question", "Blur an image spatially and show that convolution in space "
         "equals multiplication in frequency."),
        ("Implementation", "Same finite Gaussian kernel; full linear convolution "
         "with zero boundary conditions. FFT shape is image+kernel-1 on each axis. "
         "Both outputs use the same central crop only for display."),
        ("Execution results", json.dumps(metrics, indent=2)),
        ("Interpretation", "Round-off-level differences are expected. Unpadded FFT "
         "would instead perform circular convolution. One timing run is not a benchmark."),
        ("Submission still required", "Add your accessible assignment-repository URL. "
         "Record an actual working demonstration; do not claim this PDF is a recording."),
        ("Reference", SOURCES["blur"])], [out / "comparison.png"])
def thermal(cfg):
    """Classical threshold/contour branch, independent of any SAM2 mask."""
    c, out = cv(), outdir(cfg)
    a = read_image(cfg["image"])
    gray = c.cvtColor(a, c.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0, y0, x1, y1 = map(int, cfg["roi_xyxy"])
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Invalid ROI.")
    if cfg.get("palette") not in ("white_hot", "black_hot"):
        raise ValueError("Use a documented grayscale white_hot or black_hot image.")
    roi = c.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 0)
    typ = c.THRESH_BINARY if cfg["palette"] == "white_hot" else c.THRESH_BINARY_INV
    threshold, binary = c.threshold(roi, 0, 255, typ + c.THRESH_OTSU)
    element = c.getStructuringElement(c.MORPH_ELLIPSE, (3, 3))
    binary = c.morphologyEx(binary, c.MORPH_OPEN, element)
    binary = c.morphologyEx(binary, c.MORPH_CLOSE, element)
    count, labels, stats, centroids = c.connectedComponentsWithStats(binary, 8)
    sx, sy = map(int, cfg["foreground_seed_xy"])
    if not (x0 <= sx < x1 and y0 <= sy < y1):
        raise ValueError("Foreground seed must be inside ROI.")
    selected = int(labels[sy-y0, sx-x0])
    if selected == 0:
        raise ValueError("Seed is background after thresholding. Inspect palette/ROI.")
    roi_mask = (labels == selected).astype(np.uint8) * 255
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = roi_mask
    contours, _ = c.findContours(mask, c.RETR_EXTERNAL, c.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No classical contour.")
    overlay = a.copy()
    c.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    c.imwrite(str(out / "classical_mask.png"), mask)
    c.imwrite(str(out / "classical_boundary.png"), overlay)
    points = max(contours, key=c.contourArea).reshape(-1, 2)
    np.savetxt(out / "boundary_xy.csv", points, delimiter=",", header="x,y", comments="")
    metrics = {"image_sha256": fingerprint(cfg["image"]),
               "otsu_threshold": float(threshold), "area_pixels": int((mask > 0).sum()),
               "roi_xyxy": [x0, y0, x1, y1], "seed_xy": [sx, sy],
               "method": "manual ROI/seed + Otsu + morphology + connected component"}
    pictures = [out / "classical_boundary.png"]
    if cfg.get("sam2_mask"):
        other = c.imread(str(cfg["sam2_mask"]), c.IMREAD_GRAYSCALE)
        if other is None or other.shape != mask.shape:
            raise ValueError("SAM2 mask must match original image resolution.")
        A, B = mask > 0, other > 0
        union, total = int((A | B).sum()), int(A.sum() + B.sum())
        intersection = int((A & B).sum())
        metrics["sam2_iou_agreement"] = intersection / union if union else None
        metrics["sam2_dice_agreement"] = 2 * intersection / total if total else None
        metrics["comparison_caution"] = "Agreement with SAM2 is not ground-truth accuracy."
        sam_contours, _ = c.findContours(B.astype(np.uint8), c.RETR_EXTERNAL,
                                         c.CHAIN_APPROX_NONE)
        c.drawContours(overlay, sam_contours, -1, (0, 0, 255), 2)
        c.imwrite(str(out / "comparison_green_classical_red_sam2.png"), overlay)
        pictures.append(out / "comparison_green_classical_red_sam2.png")
    else:
        metrics["sam2_status"] = "NOT RUN: required comparison remains incomplete"
    save_json(out / "metrics.json", metrics)
    report(out, "Module 4: thermal animal boundary", [
        ("Data and inputs", cfg.get("source_citation", "Record the image source/license.")
         + "\nDeclared manual ROI and foreground seed. " + cfg["palette"] + " palette."),
        ("Results", json.dumps(metrics, indent=2)),
        ("Limitations", "Intensity segmentation may miss thin limbs or join touching "
         "warm objects. Inspect the contour. No exact-ground-truth claim is justified."),
        ("Reference for independent comparison", SOURCES["sam2"])], pictures)
def sam2_mask(cfg):
    """Optional separate comparator. Install SAM2 per its official repository."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    out = outdir(cfg)
    device = cfg.get("device", "cpu")
    image = np.asarray(Image.open(cfg["image"]).convert("RGB"))
    box = np.asarray(cfg["roi_xyxy"], dtype=np.float32)
    model = build_sam2(cfg["model_config"], cfg["checkpoint"], device=device)
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = np.asarray(masks).reshape(-1, image.shape[0], image.shape[1])[0]
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "sam2_mask.png")
    save_json(out / "sam2_details.json", {
        "model_config": cfg["model_config"],
        "checkpoint_sha256": fingerprint(cfg["checkpoint"]),
        "image_sha256": fingerprint(cfg["image"]),
        "prompt_box": box.tolist(),
        "predicted_score_not_accuracy": np.asarray(scores).tolist()})
def height(cfg):
    """Manually initialized same-plane reference-scale prototype."""
    c, out = cv(), outdir(cfg)
    a = read_image(cfg["image"])
    person = as_points(cfg["person_head_feet"], 2)
    reference = as_points(cfg["reference_top_bottom"], 2)
    check_points(person, a.shape[1], a.shape[0])
    check_points(reference, a.shape[1], a.shape[0])
    if not cfg.get("same_plane_and_level_camera_confirmed", False):
        raise ValueError("Same-plane geometry and level camera must be confirmed.")
    if cfg.get("calibration"):
        K, dist, size = calibration_load(cfg["calibration"])
        if a.shape[1::-1] != size:
            raise ValueError("Calibration resolution mismatch.")
        person_measure = c.undistortPoints(person[:, None, :], K, dist, P=K).reshape(-1, 2)
        ref_measure = c.undistortPoints(reference[:, None, :], K, dist, P=K).reshape(-1, 2)
    else:
        person_measure, ref_measure = person, reference
    span_person = float(np.linalg.norm(person_measure[1] - person_measure[0]))
    span_ref = float(np.linalg.norm(ref_measure[1] - ref_measure[0]))
    length = float(cfg["reference_height_m"])
    truth = float(cfg["true_person_height_m"])
    if min(span_person, span_ref, length, truth) <= 0:
        raise ValueError("All lengths/spans must be positive.")
    estimate = length * span_person / span_ref
    for p, color in [(person, (0, 255, 0)), (reference, (0, 0, 255))]:
        c.line(a, tuple(np.rint(p[0]).astype(int)),
               tuple(np.rint(p[1]).astype(int)), color, 3)
    c.putText(a, "Estimated height %.3f m" % estimate,
              (15, 40), c.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
    c.imwrite(str(out / "height_measurement.png"), a)
    result = {"estimated_height_m": estimate, "ground_truth_m": truth,
              "reference_height_m": length, "person_span_px": span_person,
              "reference_span_px": span_ref, "signed_error_m": estimate-truth,
              "absolute_percentage_error": 100*abs(estimate-truth)/truth,
              "input_sha256": fingerprint(cfg["image"]),
              "manual_inputs": "person endpoints and reference endpoints"}
    save_json(out / "metrics.json", result)
    report(out, "Week 5: approximate height prototype", [
        ("Method", "Multiply measured reference length by the ratio of image spans. "
         "Same depth, parallel plane and negligible remaining distortion are assumed."),
        ("Results", json.dumps(result, indent=2)),
        ("Limitations", "Manual endpoint initialization. Not general automatic person "
         "detection. Plane mismatch, tilt, footwear and posture can change the estimate.")],
        [out / "height_measurement.png"])
def bilinear(image, x, y):
    """Sample a 2D grayscale image at subpixel coordinates; fail outside its domain."""
    h, w = image.shape
    x, y = np.asarray(x), np.asarray(y)
    if np.any(x < 0) or np.any(y < 0) or np.any(x >= w-1) or np.any(y >= h-1):
        raise ValueError("Bilinear coordinates outside valid interpolation domain.")
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = x-i, y-j
    return ((1-a)*(1-b)*image[j, i] + a*(1-b)*image[j, i+1]
            + (1-a)*b*image[j+1, i] + a*b*image[j+1, i+1])
def lk_from_scratch(first, second, points, radius=5, iterations=25):
    """Single-level iterative Lucas-Kanade using first-image gradients.
    Returns locations and validity flags. Limited to locally small motion.
    """
    A0, B0 = first.astype(float)/255, second.astype(float)/255
    gy, gx = np.gradient(A0)
    oy, ox = np.mgrid[-radius:radius+1, -radius:radius+1]
    tracked, valid = [], []
    for px, py in points:
        xx, yy = px+ox.ravel(), py+oy.ravel()
        d = np.zeros(2)
        good = True
        try:
            template = bilinear(A0, xx, yy)
            design = np.column_stack([bilinear(gx, xx, yy), bilinear(gy, xx, yy)])
            if np.linalg.matrix_rank(design) < 2 or np.linalg.cond(design) > 1e4:
                raise ValueError("Patch has insufficient two-direction texture.")
            for _ in range(iterations):
                residual = template - bilinear(B0, xx+d[0], yy+d[1])
                step = np.linalg.lstsq(design, residual, rcond=None)[0]
                d += step
                if np.linalg.norm(step) < 1e-3:
                    break
        except ValueError:
            good = False
        tracked.append([px+d[0], py+d[1]])
        valid.append(good)
    return np.asarray(tracked), np.asarray(valid)
def flow(cfg):
    """Two real >=30-second clips; dense visualization and independent tracking checks."""
    c, out = cv(), outdir(cfg)
    videos = cfg["videos"]
    if len(videos) != 2:
        raise ValueError("The assignment requires exactly two video samples here.")
    all_results = []
    for number, item in enumerate(videos, 1):
        dest = out / ("video_" + str(number))
        dest.mkdir(exist_ok=True)
        cap = c.VideoCapture(str(item["path"]))
        fps = float(cap.get(c.CAP_PROP_FPS))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Unreadable video or invalid frame rate.")
        seconds = float(item.get("seconds", 30))
        if seconds < 30:
            raise ValueError("Use at least 30 seconds per sample.")
        needed = int(math.ceil(seconds * fps))
        start_seconds = float(item.get("start_seconds", 0))
        cap.set(c.CAP_PROP_POS_MSEC, start_seconds * 1000)
        ok, previous_bgr = cap.read()
        if not ok:
            raise ValueError("No initial frame.")
        previous = c.cvtColor(previous_bgr, c.COLOR_BGR2GRAY)
        h, w = previous.shape
        writer = c.VideoWriter(str(dest / "optical_flow.mp4"),
                               c.VideoWriter_fourcc(*"mp4v"), fps, (w*2, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("MP4 writer could not open.")
        magnitude_scale = float(item.get("display_max_px_per_frame", 10))
        if magnitude_scale <= 0:
            raise ValueError("Display magnitude scale must be positive.")
        pair_index = int(item.get("validation_pair_index", 0))
        if not (0 <= pair_index < needed-1):
            raise ValueError("Validation pair must be within the requested clip.")
        validation_pair = None
        mean_magnitudes = []
        first_visual = None
        try:
            writer.write(np.hstack([previous_bgr, np.zeros_like(previous_bgr)]))
            for index in range(needed-1):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError("Video ends before the required sample duration.")
                gray = c.cvtColor(bgr, c.COLOR_BGR2GRAY)
                if gray.shape != (h, w):
                    raise ValueError("Frame resolution changed.")
                f = c.calcOpticalFlowFarneback(
                    previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                mag, angle = c.cartToPolar(f[..., 0], f[..., 1])
                hsv = np.zeros((h, w, 3), np.uint8)
                hsv[..., 0] = np.mod(angle * 90/np.pi, 180).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(mag/magnitude_scale*255, 0, 255).astype(np.uint8)
                vis = c.cvtColor(hsv, c.COLOR_HSV2BGR)
                panel = np.hstack([bgr, vis])
                c.putText(panel, "Hue=direction; brightness=0 to %.1f px/frame" % magnitude_scale,
                          (8, 24), c.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                writer.write(panel)
                if first_visual is None:
                    first_visual = panel.copy()
                    c.imwrite(str(dest / "flow_sample.png"), panel)
                mean_magnitudes.append(float(np.mean(mag)))
                if index == pair_index:
                    validation_pair = (previous.copy(), gray.copy())
                previous = gray
        finally:
            writer.release()
            cap.release()
        a, b = validation_pair
        p = as_points(item["manual_points_first"])
        truth = as_points(item["manual_points_next"], len(p))
        if len(p) < 6:
            raise ValueError("Mark at least six corresponding validation features.")
        check_points(p, w, h)
        check_points(truth, w, h)
        pred, status, _ = c.calcOpticalFlowPyrLK(
            a, b, p.astype(np.float32)[:, None, :], None,
            winSize=(21, 21), maxLevel=3,
            criteria=(c.TERM_CRITERIA_EPS + c.TERM_CRITERIA_COUNT, 30, .01))
        if pred is None:
            raise ValueError("No tracked features.")
        pred = pred.reshape(-1, 2)
        valid = status.ravel().astype(bool) & np.isfinite(pred).all(axis=1)
        original, valid_original = lk_from_scratch(a, b, p)
        endpoint = np.linalg.norm(pred-truth, axis=1)
        endpoint_original = np.linalg.norm(original-truth, axis=1)
        rows = []
        for i in range(len(p)):
            rows.append([i, *p[i], *truth[i], *pred[i], int(valid[i]),
                         endpoint[i] if valid[i] else "",
                         *original[i], int(valid_original[i]),
                         endpoint_original[i] if valid_original[i] else ""])
        with open(dest / "tracking_validation.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "x0", "y0", "manual_x1", "manual_y1", "pyrLK_x1",
                         "pyrLK_y1", "pyrLK_valid", "pyrLK_error_px",
                         "scratch_x1", "scratch_y1", "scratch_valid", "scratch_error_px"])
            wr.writerows(rows)
        overlay = c.cvtColor(b, c.COLOR_GRAY2BGR)
        for point, measured, prediction, good in zip(p, truth, pred, valid):
            c.circle(overlay, tuple(np.rint(measured).astype(int)), 5, (0, 255, 0), 1)
            if good:
                c.circle(overlay, tuple(np.rint(prediction).astype(int)), 3, (0, 0, 255), -1)
        c.imwrite(str(dest / "tracking_green_manual_red_predicted.png"), overlay)
        stats = {
            "video": item["path"], "sha256": fingerprint(item["path"]),
            "fps": fps, "processed_frames": needed, "duration_seconds": needed/fps,
            "validation_pair_relative_index": pair_index,
            "mean_flow_magnitude_px_per_frame": float(np.mean(mean_magnitudes)),
            "manual_annotations": len(p), "pyrLK_valid": int(valid.sum()),
            "pyrLK_mean_endpoint_error_px": float(endpoint[valid].mean()) if valid.any() else None,
            "scratch_valid": int(valid_original.sum()),
            "scratch_mean_endpoint_error_px": float(endpoint_original[valid_original].mean())
                if valid_original.any() else None,
            "flow_display_max_px_per_frame": magnitude_scale}
        save_json(dest / "metrics.json", stats)
        all_results.append(stats)
        report(dest, "Assignment 6: optical flow and tracking", [
            ("Method", "Dense Farneback flow visualization plus pyramidal Lucas-Kanade. "
             "A separate small-motion implementation directly solves patch least squares."),
            ("Actual execution", json.dumps(stats, indent=2)),
            ("Validation", "Manual next-frame positions are independent references; "
             "do not replace them with the tracker predictions."),
            ("Interpretation", "Hue follows image-plane motion angle; brightness uses a "
             "fixed declared magnitude scale. Optical flow is apparent 2D motion, not "
             "direct physical velocity. Compare moving-camera and moving-object cases."),
            ("Reference", SOURCES["flow"])],
            [dest / "flow_sample.png", dest / "tracking_green_manual_red_predicted.png"])
    save_json(out / "summary.json", all_results)
def pose_for_view(view, K, dist):
    """Estimate metric camera pose from a separate known calibration target."""
    c = cv()
    if "calibration_xyz_m" in view:
        world = np.asarray(view["calibration_xyz_m"], np.float64)
        image = as_points(view["calibration_xy"], len(world))
        if world.ndim != 2 or world.shape[1] != 3 or len(world) < 6:
            raise ValueError("Use >=6 known calibration target points in a common world frame.")
        ok, rvec, tvec = c.solvePnP(world, image, K, dist, flags=c.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("Pose estimation failed.")
        R, _ = c.Rodrigues(rvec)
        projected, _ = c.projectPoints(world, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2)-image)**2, axis=1))))
        provenance = "Pose fitted from a separately measured calibration target"
    else:
        R = np.asarray(view["R_world_to_camera"], float)
        tvec = np.asarray(view["t_world_to_camera_m"], float).reshape(3, 1)
        rms = None
        provenance = view.get("pose_provenance", "")
        if not provenance:
            raise ValueError("Describe how camera poses were measured/estimated.")
    if R.shape != (3, 3) or not np.allclose(R.T@R, np.eye(3), atol=1e-4):
        raise ValueError("Invalid rotation matrix.")
    if not np.isclose(np.linalg.det(R), 1, atol=1e-4):
        raise ValueError("Rotation determinant must be +1.")
    return R, tvec.reshape(3), rms, provenance
def four_view(cfg):
    """Fiducial-assisted four-view reconstruction of ordered planar boundary points."""
    c, out = cv(), outdir(cfg)
    if len(cfg["views"]) != 4:
        raise ValueError("Exactly four views are required.")
    cameras, normalized, observations, poses = [], [], [], []
    count = None
    for view in cfg["views"]:
        img = read_image(view["image"])
        K, dist, size = calibration_load(view["calibration"])
        if img.shape[1::-1] != size:
            raise ValueError("Calibration/image resolution mismatch.")
        p = as_points(view["boundary_xy"])
        if count is None:
            count = len(p)
        if len(p) != count or count < 6:
            raise ValueError("Use >=6 identical ordered boundary points in every view.")
        check_points(p, *size)
        R, t, fit, provenance = pose_for_view(view, K, dist)
        P = np.column_stack([R, t])
        n = c.undistortPoints(p[:, None, :], K, dist).reshape(-1, 2)
        cameras.append(P)
        normalized.append(n)
        observations.append((img, K, dist, p))
        poses.append({"R": R.tolist(), "t_m": t.tolist(),
                      "camera_center_world_m": (-R.T@t).tolist(),
                      "pose_fit_rms_px": fit, "provenance": provenance})
    centres = np.array([p["camera_center_world_m"] for p in poses])
    if np.max(np.linalg.norm(centres[:, None]-centres[None, :], axis=2)) < 1e-4:
        raise ValueError("Camera translation is too small for useful triangulation.")
    points = []
    for j in range(count):
        rows = []
        for P, p in zip(cameras, normalized):
            x, y = p[j]
            rows.extend([x*P[2]-P[0], y*P[2]-P[1]])
        _, singular, vh = np.linalg.svd(np.asarray(rows))
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            raise ValueError("Point at infinity/degenerate triangulation.")
        X = homogeneous[:3]/homogeneous[3]
        if any((P@np.r_[X, 1])[2] <= 0 for P in cameras):
            raise ValueError("Point is behind a camera; check correspondence/pose.")
        points.append(X)
    points = np.asarray(points)
    errors, pics = [], []
    for i, (P, (img, K, dist, p)) in enumerate(zip(cameras, observations)):
        rvec, _ = c.Rodrigues(P[:, :3])
        projected, _ = c.projectPoints(points, rvec, P[:, 3], K, dist)
        projected = projected.reshape(-1, 2)
        e = np.linalg.norm(projected-p, axis=1)
        errors.append(e)
        for measured, predicted in zip(p, projected):
            c.circle(img, tuple(np.rint(measured).astype(int)), 5, (0, 255, 0), 1)
            c.circle(img, tuple(np.rint(predicted).astype(int)), 2, (0, 0, 255), -1)
        path = out / ("view_%d_reprojection.png" % (i+1))
        c.imwrite(str(path), img)
        pics.append(path)
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points-centre, full_matrices=False)
    coordinates = (points-centre)@vh[:2].T
    residual = (points-centre)@vh[2]
    closed = np.vstack([coordinates, coordinates[0]])
    f, ax = plt.subplots(figsize=(7, 6))
    ax.plot(closed[:, 0], closed[:, 1], "-o")
    for i, p in enumerate(coordinates):
        ax.annotate(str(i), p)
    ax.set(xlabel="Plane coordinate 1 (m)", ylabel="Plane coordinate 2 (m)",
           title="Triangulated ordered boundary; plane orientation arbitrary")
    ax.axis("equal")
    f.tight_layout()
    f.savefig(out / "planar_boundary.png", dpi=160)
    plt.close(f)
    np.savetxt(out / "boundary_xyz_m.csv", points, delimiter=",",
               header="X_m,Y_m,Z_m", comments="")
    metrics = {"method": "Four-view DLT with independently estimated/supplied metric poses",
               "poses": poses, "boundary_point_count": count,
               "per_view_reprojection_rmse_px": np.sqrt(
                   np.mean(np.array(errors)**2, axis=1)).tolist(),
               "plane_fit_rms_m": float(np.sqrt(np.mean(residual**2))),
               "ordered_polygon_perimeter_m": float(np.sum(np.linalg.norm(
                   np.roll(points, -1, axis=0)-points, axis=1)))}
    save_json(out / "metrics.json", metrics)
    report(out, "Assignment 6: four-view planar reconstruction", [
        ("Method", "For each point, stack xP3-P1 and yP3-P2 over four cameras. "
         "Take the last right-singular vector and dehomogenize. Enforce positive depth."),
        ("Results", json.dumps(metrics, indent=2)),
        ("Interpretation", "Pose units establish scale. Reprojection is an internal "
         "consistency check, not independent dimensional accuracy. Sparse boundary "
         "points approximate the outline between observations. This is fiducial-assisted "
         "reconstruction; do not describe supplied poses as image-only SfM."),
        ("Reference", SOURCES["calibration"])], [out / "planar_boundary.png"] + pics)
def rectified_map(cfg):
    """Week 8. Only for truly rectified, matched-intrinsic stereo images."""
    c, out = cv(), outdir(cfg)
    if not cfg.get("rectified_equal_principal_points_confirmed", False):
        raise ValueError("Calibrate and rectify first; raw images are not accepted.")
    K = np.asarray(cfg["rectified_K"], float)
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid rectified intrinsic matrix.")
    baseline = float(cfg["baseline_m"])
    if baseline <= 0:
        raise ValueError("Baseline must be a measured positive distance.")
    floor_R = np.asarray(cfg["R_camera_to_floor"], float)
    floor_t = np.asarray(cfg.get("t_camera_to_floor_m", [0, 0, 0]), float)
    if (floor_R.shape != (3, 3) or
        not np.allclose(floor_R.T@floor_R, np.eye(3), atol=1e-4) or
        not np.isclose(np.linalg.det(floor_R), 1, atol=1e-4)):
        raise ValueError("Provide a valid measured camera-to-floor rotation.")
    rows = []
    for obj in cfg["objects"]:
        if obj["kind"] not in ("table", "chair"):
            raise ValueError("Object kind must be table or chair.")
        ul, vl = map(float, obj["left_xy"])
        ur, vr = map(float, obj["right_xy"])
        if abs(vl-vr) > float(cfg.get("vertical_tolerance_px", 2)):
            raise ValueError("Vertical mismatch: rectify or correct correspondence.")
        disparity = ul-ur
        if disparity <= float(cfg.get("minimum_disparity_px", 1)):
            raise ValueError("Disparity is nonpositive or too small for stable depth.")
        Z = K[0, 0]*baseline/disparity
        X = (ul-K[0, 2])*Z/K[0, 0]
        Y = (vl-K[1, 2])*Z/K[1, 1]
        floor = floor_R@np.array([X, Y, Z])+floor_t
        rows.append({"id": obj["id"], "kind": obj["kind"],
                     "disparity_px": disparity, "camera_X_m": X, "camera_Y_m": Y,
                     "camera_Z_m": Z, "floor_X_m": float(floor[0]),
                     "floor_Y_m": float(floor[1]), "floor_Z_m": float(floor[2]),
                     "reference_point": obj["reference_point"]})
    if not rows:
        raise ValueError("List every required classroom table and chair.")
    f, ax = plt.subplots(figsize=(8, 7))
    for kind, color in [("table", "red"), ("chair", "blue")]:
        group = [r for r in rows if r["kind"] == kind]
        if group:
            ax.scatter([r["floor_X_m"] for r in group],
                       [r["floor_Y_m"] for r in group], c=color, label=kind)
            for r in group:
                ax.annotate(r["id"], (r["floor_X_m"], r["floor_Y_m"]))
    ax.set(xlabel="Floor X (m)", ylabel="Floor Y (m)",
           title="Classroom layout from stereo; tables red, chairs blue")
    ax.axis("equal")
    ax.legend()
    ax.grid(alpha=.3)
    f.tight_layout()
    f.savefig(out / "classroom_map.png", dpi=160)
    plt.close(f)
    with open(out / "object_coordinates.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(out, "Week 8: stereo classroom map", [
        ("Inputs", "Measured baseline: %.4f m. Objects: %d. Floor rotation and "
         "manual reference points are explicitly supplied." % (baseline, len(rows))),
        ("Equations", "Z=fx*B/disparity; X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy. "
         "Transform into the measured floor coordinate frame and plot its first two axes."),
        ("Limitations", "Correct only under the declared rectified-camera assumptions. "
         "A floor map requires true camera-to-floor orientation. Check that every table "
         "and chair is represented. Depth sensitivity grows as disparity decreases."),
        ("Source", SOURCES["calibration"])], [out / "classroom_map.png"])
def stereo(cfg):
    """Final week: F estimation, calibrated E/R recovery, scaled triangulation.
    Initially unknown relative pose is estimated. Intrinsics and metric baseline
    are still required; otherwise the physical distance is unidentifiable.
    """
    c, out = cv(), outdir(cfg)
    left, right = read_image(cfg["left_image"]), read_image(cfg["right_image"])
    K1, d1, s1 = calibration_load(cfg["left_calibration"])
    K2, d2, s2 = calibration_load(cfg["right_calibration"])
    if left.shape[1::-1] != s1 or right.shape[1::-1] != s2:
        raise ValueError("Image/calibration resolution mismatch.")
    p1 = as_points(cfg["matched_left_xy"])
    p2 = as_points(cfg["matched_right_xy"], len(p1))
    if len(p1) < 12:
        raise ValueError("Use >=12 accurate spatially distributed correspondences.")
    check_points(p1, *s1)
    check_points(p2, *s2)
    u1 = c.undistortPoints(p1[:, None, :], K1, d1, P=K1).reshape(-1, 2)
    u2 = c.undistortPoints(p2[:, None, :], K2, d2, P=K2).reshape(-1, 2)
    F, mask = c.findFundamentalMat(u1, u2, c.FM_RANSAC, 1.5, .999)
    if F is None or F.shape != (3, 3) or mask is None:
        raise ValueError("Fundamental matrix estimation failed.")
    keep = mask.ravel().astype(bool)
    if keep.sum() < 8:
        raise ValueError("Too few epipolar inliers.")
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U@np.diag(S)@Vt
    F /= np.linalg.norm(F)
    E_raw = K2.T@F@K1
    U, S, Vt = np.linalg.svd(E_raw)
    equal = (S[0]+S[1])/2
    E = U@np.diag([equal, equal, 0])@Vt
    n1 = c.undistortPoints(p1[keep, None, :], K1, d1).reshape(-1, 2)
    n2 = c.undistortPoints(p2[keep, None, :], K2, d2).reshape(-1, 2)
    positive, R, direction, pose_mask = c.recoverPose(E, n1, n2, np.eye(3))
    if positive < 6:
        raise ValueError("Too few positive-depth points. Check geometry/parallax.")
    B = float(cfg["baseline_m"])
    if B <= 0:
        raise ValueError("A measured baseline is required for metric scale.")
    t = direction.reshape(3)/np.linalg.norm(direction)*B
    object_left = as_points([cfg["object_left_xy"]], 1)
    object_right = as_points([cfg["object_right_xy"]], 1)
    check_points(object_left, *s1)
    check_points(object_right, *s2)
    q1 = c.undistortPoints(object_left[:, None, :], K1, d1).reshape(-1, 2)
    q2 = c.undistortPoints(object_right[:, None, :], K2, d2).reshape(-1, 2)
    P1 = np.column_stack([np.eye(3), np.zeros(3)])
    P2 = np.column_stack([R, t])
    Xh = c.triangulatePoints(P1, P2, q1.T, q2.T)[:, 0]
    if abs(Xh[3]) < 1e-10:
        raise ValueError("Object point triangulates at infinity.")
    X = Xh[:3]/Xh[3]
    if X[2] <= 0 or (R@X+t)[2] <= 0:
        raise ValueError("Object point behind camera.")
    distance = float(np.linalg.norm(X))
    truth = float(cfg["ground_truth_camera_to_point_range_m"])
    if truth <= 0:
        raise ValueError("Ground-truth range must be positive.")
    # Project back into the original distorted images for interpretable residuals.
    back1, _ = c.projectPoints(X[None, :], np.zeros(3), np.zeros(3), K1, d1)
    rvec, _ = c.Rodrigues(R)
    back2, _ = c.projectPoints(X[None, :], rvec, t, K2, d2)
    errors = [float(np.linalg.norm(back1.reshape(2)-object_left[0])),
              float(np.linalg.norm(back2.reshape(2)-object_right[0]))]
    F_consistent = np.linalg.inv(K2).T@E@np.linalg.inv(K1)
    F_consistent /= np.linalg.norm(F_consistent)
    metrics = {"F_rank2_from_matches": F.tolist(),
               "E_constrained": E.tolist(), "F_consistent_with_E": F_consistent.tolist(),
               "R_camera1_to_camera2": R.tolist(), "t_camera1_to_camera2_m": t.tolist(),
               "baseline_m": B, "epipolar_inliers": int(keep.sum()),
               "positive_depth_inliers": int(positive),
               "object_XYZ_camera1_m": X.tolist(), "optical_depth_Z_m": float(X[2]),
               "euclidean_range_m": distance, "ground_truth_range_m": truth,
               "absolute_range_error_m": abs(distance-truth),
               "absolute_percentage_error": 100*abs(distance-truth)/truth,
               "object_reprojection_error_px": errors,
               "warning": "F-only geometry is not metric; intrinsics and baseline were supplied."}
    for img, p, side in [(left, object_left[0], "left"), (right, object_right[0], "right")]:
        c.circle(img, tuple(np.rint(p).astype(int)), 9, (0, 255, 0), 3)
        c.putText(img, "Estimated range %.3fm; measured %.3fm" % (distance, truth),
                  (10, 35), c.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        c.imwrite(str(out / (side + "_annotated.png")), img)
    pictures = [out / "left_annotated.png", out / "right_annotated.png"]
    if cfg.get("setup_photo"):
        pictures.append(Path(cfg["setup_photo"]))
    else:
        metrics["setup_photo_status"] = "MISSING: required in final assignment report"
    save_json(out / "metrics.json", metrics)
    report(out, "Final week: stereo distance estimation", [
        ("Method", "Estimate F on undistorted pixel correspondences. Form E=K2^T F K1, "
         "enforce its singular values, recover relative R/t direction using cheirality, "
         "then apply measured baseline scale and triangulate the object."),
        ("Execution results", json.dumps(metrics, indent=2)),
        ("Limitations", "Planar matches, pure rotation, low parallax and poor calibration "
         "can make reconstruction unstable. Use textured matches across multiple depths. "
         "Range and axial depth are different quantities; compare the same definition "
         "with ground truth. Reprojection consistency alone does not establish accuracy."),
        ("Reference", SOURCES["calibration"])], pictures)
def pick_points(cfg):
    """Click ordered points; Enter saves, U undoes, Esc cancels.
    Display is downscaled only for viewing; saved coordinates remain original.
    """
    c, out = cv(), outdir(cfg)
    a = read_image(cfg["image"])
    scale = min(1.0, 1200/a.shape[1], 800/a.shape[0])
    base = c.resize(a, (round(a.shape[1]*scale), round(a.shape[0]*scale)))
    points = []
    name = "Click points in order; Enter=save, U=undo, Esc=cancel"
    def click(event, x, y, flags, userdata):
        if event == c.EVENT_LBUTTONDOWN:
            points.append([x/scale, y/scale])
    c.namedWindow(name)
    c.setMouseCallback(name, click)
    try:
        while True:
            show = base.copy()
            for i, (x, y) in enumerate(points):
                px = (round(x*scale), round(y*scale))
                c.circle(show, px, 4, (0, 255, 0), -1)
                c.putText(show, str(i), px, c.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            c.imshow(name, show)
            key = c.waitKey(20) & 255
            if key in (10, 13):
                if len(points) < int(cfg.get("minimum_points", 1)):
                    continue
                break
            if key in (ord("u"), ord("U")) and points:
                points.pop()
            if key == 27:
                raise RuntimeError("Point selection cancelled.")
    finally:
        c.destroyAllWindows()
    save_json(out / "points.json", {"image": cfg["image"], "points_xy": points})
def frames(cfg):
    """Extract the same consecutive frame pair used by flow validation."""
    c, out = cv(), outdir(cfg)
    cap = c.VideoCapture(str(cfg["video"]))
    fps = float(cap.get(c.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot read video or frame rate.")
    cap.set(c.CAP_PROP_POS_MSEC, float(cfg.get("start_seconds", 0))*1000)
    index = int(cfg.get("pair_index", 0))
    if index < 0:
        raise ValueError("Pair index cannot be negative.")
    try:
        for _ in range(index+1):
            ok, a = cap.read()
            if not ok:
                raise ValueError("Requested frame not present.")
        ok, b = cap.read()
        if not ok:
            raise ValueError("Next frame not present.")
    finally:
        cap.release()
    c.imwrite(str(out / "first.png"), a)
    c.imwrite(str(out / "next.png"), b)
    save_json(out / "frame_pair.json", {"source": cfg["video"], "fps": fps,
              "start_seconds": float(cfg.get("start_seconds", 0)), "pair_index": index})
COMMANDS = {
    "blur": blur, "calibrate": calibrate, "dimensions": dimensions,
    "thermal": thermal, "sam2": sam2_mask, "height": height,
    "flow": flow, "four-view": four_view, "stereo-map": rectified_map,
    "stereo": stereo, "points": pick_points, "frames": frames
}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, help="JSON config (required except blur)")
    parser.add_argument("--out", default="outputs/module3",
                        help="Output folder for the default blur demonstration")
    args = parser.parse_args()
    if args.config:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.command == "blur":
        cfg = {"out": args.out}
    else:
        parser.error("--config is required for this command.")
    out = outdir(cfg)
    # Preserve exact invocation inputs and runtime versions for reproducibility.
    save_json(out / "run_config.json", cfg)
    save_json(out / "runtime.json", {"python": platform.python_version(),
                                   "numpy": np.__version__,
                                   "platform": platform.platform()})
    COMMANDS[args.command](cfg)
    print("Completed:", args.command, "Outputs:", out.resolve())
    print("Inspect results and remaining assignment requirements before submission.")
if __name__ == "__main__":
    main()

