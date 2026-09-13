"""Experimental regional source sharpening. Not connected to production presets."""
import cv2
import numpy as np


def analyze(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4:
        raise ValueError('Expected a BGR uint8 image of at least 4x4')
    y = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    h, w = y.shape
    residual = cv2.filter2D(y, -1, np.array([[1, -1], [-1, 1]], np.float32)/2)
    gradient = np.abs(cv2.Sobel(y, -1, 1, 0)) + np.abs(cv2.Sobel(y, -1, 0, 1))
    ys = np.linspace(0, h, max(1, (h+31)//32)+1, dtype=int)
    xs = np.linspace(0, w, max(1, (w+31)//32)+1, dtype=int)
    estimates = np.zeros((len(ys)-1, len(xs)-1), np.float32)
    for row, (top, bottom) in enumerate(zip(ys[:-1], ys[1:])):
        for col, (left, right) in enumerate(zip(xs[:-1], xs[1:])):
            g = gradient[top:bottom, left:right]
            r = residual[top:bottom, left:right]
            low_gradient = g <= np.percentile(g, 60)
            estimates[row, col] = np.median(np.abs(r[low_gradient])) / .6745
    noise = cv2.resize(estimates, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = cv2.GaussianBlur(noise, (0, 0), 2)
    smooth = cv2.GaussianBlur(y, (0, 0), .7)
    average = cv2.boxFilter(smooth, -1, (5, 5))
    variance = np.maximum(cv2.boxFilter(smooth*smooth, -1, (5, 5))-average*average, 0)
    signal = np.maximum(variance-.2*noise*noise, 0)
    support = signal/(signal+noise*noise+1)
    medium = smooth-cv2.GaussianBlur(smooth, (0, 0), 1.2)
    fine = y-smooth
    ratio = cv2.GaussianBlur(np.abs(fine), (0, 0), 1)/(cv2.GaussianBlur(np.abs(medium), (0, 0), 1)+1)
    protection = np.clip(2-2*ratio, 0, 1)
    return {'luminance': y, 'noise_sigma': noise, 'noise_confidence': noise/(1+noise),
            'detail_support': support, 'edge_protection': protection, 'fine': fine, 'medium': medium}


def sharpen(image, strength=.5, upscale=1, variant='regional_structure', maps=None):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4:
        raise ValueError('Expected a BGR uint8 image of at least 4x4')
    if variant not in ('regional_noise', 'regional_structure', 'guided_structure'):
        raise ValueError('Unknown experimental variant')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite in [0, 1]')
    if not isinstance(upscale, (int, np.integer)) or upscale < 1:
        raise ValueError('upscale must be a positive integer')
    maps = analyze(image) if maps is None else maps
    for key in ('luminance', 'noise_sigma', 'detail_support', 'edge_protection', 'fine', 'medium'):
        value = maps[key]
        if value.shape != image.shape[:2] or not np.isfinite(value).all():
            raise ValueError(f'Invalid map: {key}')
    noise, y = maps['noise_sigma'], maps['luminance']
    if np.any(noise < 0) or any(np.any((maps[k] < 0) | (maps[k] > 1)) for k in ('detail_support', 'edge_protection')):
        raise ValueError('Invalid map range')
    core = lambda d, t: np.sign(d)*np.maximum(np.abs(d)-t, 0)
    detail = 2.8*core(maps['medium'], np.maximum(.25, noise*.35)) + .8*core(maps['fine'], np.maximum(.5, noise*1.5))
    if variant == 'guided_structure':
        mean = cv2.boxFilter(y, -1, (5, 5))
        variance = np.maximum(cv2.boxFilter(y*y, -1, (5, 5))-mean*mean, 0)
        a = variance/(variance+(noise+2)**2)
        base = cv2.boxFilter(a, -1, (5, 5))*y + cv2.boxFilter((1-a)*mean, -1, (5, 5))
        detail = 3.6*core(y-base, np.maximum(.5, noise))
    gain = strength/(1+noise)
    if variant != 'regional_noise':
        gain = gain*maps['detail_support']*maps['edge_protection']
    delta = np.clip(gain*detail, -24*strength, 24*strength)
    kernel = np.ones((3, 3), np.uint8)
    delta = np.clip(delta, cv2.erode(y, kernel)-y, cv2.dilate(y, kernel)-y)
    h, w = y.shape
    output = cv2.resize(image, (w*upscale, h*upscale), interpolation=cv2.INTER_LANCZOS4) if upscale != 1 else image
    if upscale != 1:
        delta = cv2.resize(delta, (w*upscale, h*upscale), interpolation=cv2.INTER_LINEAR)
    output = output.astype(np.float32)
    out_y = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    lower = np.maximum(cv2.erode(out_y, kernel)-out_y, -output.min(axis=2))
    upper = np.minimum(cv2.dilate(out_y, kernel)-out_y, 255-output.max(axis=2))
    return (output+np.rint(np.clip(delta, lower, upper))[:, :, None]).astype(np.uint8)
