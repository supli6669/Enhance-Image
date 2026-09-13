"""Experimental source-only, regularized inverse filtering; no production routing."""
import cv2
import numpy as np


def validate(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise ValueError('Expected BGR uint8 image, at least 16x16')


def characterize(image):
    """Measure isolated monotonic edge profiles; not a calibrated blur classifier."""
    validate(image)
    y = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    residual = cv2.filter2D(y, -1, np.array([[1, -1], [-1, 1]], np.float32)/2)
    noise = float(np.median(np.abs(residual[1:-1, 1:-1]))/.6745)
    gx = cv2.Sobel(y, -1, 1, 0)
    gy = cv2.Sobel(y, -1, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    maxima = (magnitude == cv2.dilate(magnitude, np.ones((3, 3), np.uint8))) & (magnitude >= 40)
    maxima[:6] = False; maxima[-6:] = False
    maxima[:, :6] = False; maxima[:, -6:] = False
    py, px = np.where(maxima)
    if not len(px):
        return {'noise_sigma': noise, 'valid_edges': 0, 'edge_sigma_median': None}
    order = np.argsort(magnitude[py, px], kind='stable')[-128:]
    px, py = px[order], py[order]
    norms = np.maximum(magnitude[py, px], 1)
    offsets = np.arange(-4, 4.01, .5, dtype=np.float32)
    map_x = px[:, None].astype(np.float32)+gx[py, px, None]/norms[:, None]*offsets
    map_y = py[:, None].astype(np.float32)+gy[py, px, None]/norms[:, None]*offsets
    profiles = cv2.remap(y, map_x, map_y, cv2.INTER_LINEAR)
    diff = np.diff(profiles, axis=1)
    signed = diff.sum(axis=1)
    total = np.abs(diff).sum(axis=1)
    valid = (np.abs(signed) >= 20) & (np.abs(signed)/(total+1e-6) >= .95)
    weight = np.abs(diff[valid])
    if not len(weight):
        return {'noise_sigma': noise, 'valid_edges': 0, 'edge_sigma_median': None}
    positions = (offsets[:-1]+offsets[1:])/2
    means = (weight*positions).sum(axis=1)/weight.sum(axis=1)
    variance = (weight*(positions-means[:, None])**2).sum(axis=1)/weight.sum(axis=1)
    return {'noise_sigma': noise, 'valid_edges': int(len(weight)),
            'edge_sigma_median': float(np.median(np.sqrt(variance)))}


def deblur(image, sigma=.6, regularization=.02, strength=.5, upscale=1):
    """Try a specified Gaussian PSF; kernel is NOT inferred from the reference."""
    validate(image)
    if any(not np.isfinite(v) for v in (sigma, regularization, strength)):
        raise ValueError('Controls must be finite')
    if not .1 <= sigma <= 2 or not .001 <= regularization <= 1 or not 0 <= strength <= 1:
        raise ValueError('Controls outside bounded experiment range')
    if not isinstance(upscale, (int, np.integer)) or upscale < 1:
        raise ValueError('upscale must be a positive integer')
    h, w = image.shape[:2]
    base = cv2.resize(image, (w*upscale, h*upscale), interpolation=cv2.INTER_LANCZOS4) if upscale != 1 else image
    if strength == 0:
        return base
    source = image.astype(np.float32)
    y = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Reflect padding prevents circular DFT boundaries from wrapping into image edges.
    padded = cv2.copyMakeBorder(y, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    kernel = cv2.getGaussianKernel(13, sigma, cv2.CV_32F)
    psf = np.zeros_like(padded)
    psf[:13, :13] = kernel@kernel.T
    psf = np.roll(psf, (-6, -6), axis=(0, 1))
    transfer = cv2.dft(psf, flags=cv2.DFT_COMPLEX_OUTPUT)[:, :, 0]
    inverse = np.clip((1+regularization)*transfer/(transfer**2+regularization), 0, 2)
    spectrum = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    restored = cv2.idft(spectrum*inverse[:, :, None], flags=cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)[16:-16, 16:-16]
    delta = strength*(restored-y)
    delta = np.sign(delta)*np.maximum(np.abs(delta)-.25, 0)
    delta = np.clip(delta, -16, 16)
    kernel3 = np.ones((3, 3), np.uint8)
    delta = np.clip(delta, cv2.erode(y, kernel3)-y, cv2.dilate(y, kernel3)-y)
    if upscale != 1:
        delta = cv2.resize(delta, (w*upscale, h*upscale), interpolation=cv2.INTER_LINEAR)
    out = base.astype(np.float32)
    out_y = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    low = np.maximum(cv2.erode(out_y, kernel3)-out_y, -out.min(axis=2))
    high = np.minimum(cv2.dilate(out_y, kernel3)-out_y, 255-out.max(axis=2))
    return (out+np.rint(np.clip(delta, low, high))[:, :, None]).astype(np.uint8)
