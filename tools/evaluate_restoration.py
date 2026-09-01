"""Quantitative and perceptual evaluation of face restoration models on fixed benchmarks.

Calculates:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- ArcFace Cosine Identity Similarity (512-dim face recognition embedding)
- Per-sample and aggregate latency (ms)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

# Ensure project and tools directories are on sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_DIR / "tools"
for p in (str(PROJECT_DIR), str(TOOLS_DIR), str(PROJECT_DIR / "models" / "CodeFormer")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import LocalAIEnhancerPipeline

REQUIRED_COLUMNS = {"id", "input_path", "reference_path", "category", "notes"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    input_path: Path
    reference_path: Optional[Path]
    category: str
    notes: str


@dataclass
class SampleMetricResult:
    sample_id: str
    category: str
    latency_ms: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    identity_similarity: Optional[float] = None
    notes: str = ""


def _resolve_path(manifest_path: Path, raw_path: str) -> Optional[Path]:
    raw_path = raw_path.strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def load_manifest(manifest_path: Path) -> List[BenchmarkSample]:
    """Load and validate a benchmark manifest without opening its images."""
    if not manifest_path.is_file():
        raise ValueError(f"Benchmark manifest was not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        samples: List[BenchmarkSample] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            category = (row.get("category") or "").strip()
            input_path = _resolve_path(manifest_path, row.get("input_path") or "")
            reference_path = _resolve_path(manifest_path, row.get("reference_path") or "")
            if not sample_id or not category or input_path is None:
                raise ValueError(f"Manifest row {row_number} needs id, input_path and category.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate benchmark id: {sample_id}")
            if input_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported input image type for {sample_id}: {input_path.suffix}")
            if reference_path and reference_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported reference image type for {sample_id}: {reference_path.suffix}")
            seen_ids.add(sample_id)
            samples.append(BenchmarkSample(sample_id, input_path, reference_path, category, row.get("notes") or ""))

    if not samples:
        raise ValueError("Benchmark manifest has no samples.")
    return samples


def validate_files(samples: Iterable[BenchmarkSample]) -> Tuple[int, int]:
    """Return input/reference counts, failing closed for missing benchmark data."""
    input_count = 0
    reference_count = 0
    for sample in samples:
        if not sample.input_path.is_file():
            raise ValueError(f"Missing input for {sample.sample_id}: {sample.input_path}")
        input_count += 1
        if sample.reference_path:
            if not sample.reference_path.is_file():
                raise ValueError(f"Missing reference for {sample.sample_id}: {sample.reference_path}")
            reference_count += 1
    return input_count, reference_count


class BenchmarkEvaluator:
    def __init__(self, device: str = "cpu", enable_lpips: bool = True, enable_identity: bool = True):
        self.device = torch.device(device)
        self.enable_lpips = enable_lpips
        self.enable_identity = enable_identity
        
        # Initialize LPIPS model
        self.lpips_fn = None
        if self.enable_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
                self.lpips_fn.eval()
            except Exception as e:
                print(f"[BenchmarkEvaluator] Warning: Failed to load LPIPS model: {e}")
                self.enable_lpips = False

        # Initialize ArcFace Identity model
        self.arcface_fn = None
        if self.enable_identity:
            try:
                from facexlib.recognition.arcface_arch import Backbone
                self.arcface_fn = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se").to(self.device)
                arcface_weights = PROJECT_DIR / "weights" / "facelib" / "recognition_arcface_ir_se50.pth"
                if arcface_weights.exists():
                    self.arcface_fn.load_state_dict(torch.load(str(arcface_weights), map_location=self.device), strict=True)
                    self.arcface_fn.eval()
                else:
                    print(f"[BenchmarkEvaluator] Warning: ArcFace weights not found at {arcface_weights}")
                    self.enable_identity = False
            except Exception as e:
                print(f"[BenchmarkEvaluator] Warning: Failed to load ArcFace model: {e}")
                self.enable_identity = False

    def evaluate_pair(self, restored_bgr: np.ndarray, reference_bgr: np.ndarray) -> Dict[str, float]:
        """Compute PSNR, SSIM, LPIPS, and ArcFace Identity metrics between restored and reference images."""
        metrics: Dict[str, float] = {}
        
        # Ensure dimensions match for pixel-wise metrics
        if restored_bgr.shape != reference_bgr.shape:
            ref_h, ref_w = reference_bgr.shape[:2]
            restored_bgr = cv2.resize(restored_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LANCZOS4)

        # 1. PSNR & SSIM
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        
        psnr_val = calculate_psnr(reference_rgb, restored_rgb, data_range=255)
        ssim_val = calculate_ssim(reference_rgb, restored_rgb, channel_axis=2, data_range=255)
        metrics["psnr"] = round(float(psnr_val), 4)
        metrics["ssim"] = round(float(ssim_val), 4)

        # Prepare normalized tensors [-1, 1] for LPIPS and ArcFace
        t_res = torch.from_numpy(restored_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 127.5 - 1.0
        t_ref = torch.from_numpy(reference_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 127.5 - 1.0

        # 2. LPIPS
        if self.enable_lpips and self.lpips_fn is not None:
            with torch.no_grad():
                lpips_val = self.lpips_fn(t_res, t_ref).item()
                metrics["lpips"] = round(float(lpips_val), 4)

        # 3. ArcFace Identity Cosine Similarity
        if self.enable_identity and self.arcface_fn is not None:
            with torch.no_grad():
                # Resize to 112x112 for ArcFace backbone
                t_res_112 = F.interpolate(t_res, (112, 112), mode="bilinear", align_corners=False)
                t_ref_112 = F.interpolate(t_ref, (112, 112), mode="bilinear", align_corners=False)
                
                feat_res = F.normalize(self.arcface_fn(t_res_112), p=2, dim=1)
                feat_ref = F.normalize(self.arcface_fn(t_ref_112), p=2, dim=1)
                
                cos_sim = torch.sum(feat_res * feat_ref, dim=1).item()
                metrics["identity_similarity"] = round(float(cos_sim), 4)

        return metrics


def run_evaluation(
    manifest_path: Path,
    device: str = "cpu",
    limit: Optional[int] = None,
    save_images_dir: Optional[Path] = None,
    preset: str = "portrait",
    fidelity_w: float = 0.6,
    upscale: int = 1,
) -> Tuple[List[SampleMetricResult], Dict[str, Any]]:
    """Execute full evaluation across benchmark samples using LocalAIEnhancerPipeline."""
    samples = load_manifest(manifest_path)
    validate_files(samples)
    
    if limit is not None and limit > 0:
        samples = samples[:limit]

    print(f"\n==================================================================")
    print(f"   STARTING RESTORATION BENCHMARK EVALUATION ({len(samples)} samples)   ")
    print(f"==================================================================")
    print(f"Device: {device} | Preset: {preset} | w: {fidelity_w} | Upscale: {upscale}")

    pipeline = LocalAIEnhancerPipeline(device=device)
    evaluator = BenchmarkEvaluator(device=device)

    if save_images_dir:
        save_images_dir.mkdir(parents=True, exist_ok=True)

    results: List[SampleMetricResult] = []
    
    for idx, sample in enumerate(samples, start=1):
        img_in = cv2.imread(str(sample.input_path))
        if img_in is None:
            raise ValueError(f"Failed to read image: {sample.input_path}")
            
        t0 = time.time()
        enhanced_bgr = pipeline.process_image(
            img_in,
            w=fidelity_w,
            upscale=upscale,
            face_upsample=False,
            blend_softness=0.5,
            detection_model="retinaface_mobile0.25",
            preset_mode=preset,
            enable_eyes=True,
            enable_lips=True,
            enable_skin=True,
            sharpen_amount=0.15,
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        metric_entry = SampleMetricResult(
            sample_id=sample.sample_id,
            category=sample.category,
            latency_ms=round(elapsed_ms, 2),
            notes=sample.notes,
        )

        if sample.reference_path and sample.reference_path.is_file():
            img_ref = cv2.imread(str(sample.reference_path))
            if img_ref is not None:
                pair_metrics = evaluator.evaluate_pair(enhanced_bgr, img_ref)
                metric_entry.psnr = pair_metrics.get("psnr")
                metric_entry.ssim = pair_metrics.get("ssim")
                metric_entry.lpips = pair_metrics.get("lpips")
                metric_entry.identity_similarity = pair_metrics.get("identity_similarity")

        if save_images_dir:
            out_file = save_images_dir / f"{sample.sample_id}_enhanced.png"
            cv2.imwrite(str(out_file), enhanced_bgr)

        results.append(metric_entry)
        
        status_line = f"[{idx}/{len(samples)}] {sample.sample_id} ({sample.category}): {elapsed_ms:.1f}ms"
        if metric_entry.psnr is not None:
            status_line += f" | PSNR: {metric_entry.psnr:.2f}dB | SSIM: {metric_entry.ssim:.4f}"
        if metric_entry.identity_similarity is not None:
            status_line += f" | ArcFace ID: {metric_entry.identity_similarity:.4f}"
        print(status_line)

    # Compute Aggregate Metrics
    summary = compute_summary(results)
    return results, summary


def compute_summary(results: List[SampleMetricResult]) -> Dict[str, Any]:
    """Compute mean metrics overall and grouped by category."""
    categories: Dict[str, List[SampleMetricResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    def calc_group_stats(group_items: List[SampleMetricResult]) -> Dict[str, Any]:
        count = len(group_items)
        avg_lat = sum(r.latency_ms for r in group_items) / count if count else 0.0
        
        psnrs = [r.psnr for r in group_items if r.psnr is not None]
        ssims = [r.ssim for r in group_items if r.ssim is not None]
        lpips_list = [r.lpips for r in group_items if r.lpips is not None]
        id_sims = [r.identity_similarity for r in group_items if r.identity_similarity is not None]

        return {
            "sample_count": count,
            "mean_latency_ms": round(avg_lat, 2),
            "mean_psnr": round(float(np.mean(psnrs)), 4) if psnrs else None,
            "mean_ssim": round(float(np.mean(ssims)), 4) if ssims else None,
            "mean_lpips": round(float(np.mean(lpips_list)), 4) if lpips_list else None,
            "mean_identity_similarity": round(float(np.mean(id_sims)), 4) if id_sims else None,
        }

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(results),
        "overall": calc_group_stats(results),
        "categories": {cat: calc_group_stats(items) for cat, items in categories.items()},
    }
    return summary


def print_markdown_report(summary: Dict[str, Any]) -> None:
    """Print clean formatted Markdown report card of benchmark results."""
    print("\n==================================================================")
    print("                    BENCHMARK SCORE CARD REPORT                   ")
    print("==================================================================")
    
    header = "| Category | Samples | Latency (ms) | Mean PSNR (dB) | Mean SSIM | Mean LPIPS | ArcFace ID Sim |"
    divider = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(divider)
    
    for cat, stats in summary.get("categories", {}).items():
        psnr_str = f"{stats['mean_psnr']:.2f}" if stats['mean_psnr'] is not None else "N/A"
        ssim_str = f"{stats['mean_ssim']:.4f}" if stats['mean_ssim'] is not None else "N/A"
        lpips_str = f"{stats['mean_lpips']:.4f}" if stats['mean_lpips'] is not None else "N/A"
        id_str = f"{stats['mean_identity_similarity']:.4f}" if stats['mean_identity_similarity'] is not None else "N/A"
        
        row = f"| {cat} | {stats['sample_count']} | {stats['mean_latency_ms']:.1f} | {psnr_str} | {ssim_str} | {lpips_str} | {id_str} |"
        print(row)
        
    ov = summary.get("overall", {})
    psnr_ov = f"{ov['mean_psnr']:.2f}" if ov['mean_psnr'] is not None else "N/A"
    ssim_ov = f"{ov['mean_ssim']:.4f}" if ov['mean_ssim'] is not None else "N/A"
    lpips_ov = f"{ov['mean_lpips']:.4f}" if ov['mean_lpips'] is not None else "N/A"
    id_ov = f"{ov['mean_identity_similarity']:.4f}" if ov['mean_identity_similarity'] is not None else "N/A"
    
    print("| **OVERALL TOTAL** | **" + str(ov['sample_count']) + "** | **" + f"{ov['mean_latency_ms']:.1f}" + "** | **" + psnr_ov + "** | **" + ssim_ov + "** | **" + lpips_ov + "** | **" + id_ov + "** |")
    print("==================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face restoration quality on fixed benchmark dataset.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "benchmarks" / "manifest.csv", help="CSV manifest file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest structure and paths without running model inference.")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute evaluation on ('cpu' or 'cuda').")
    parser.add_argument("--preset", type=str, default="portrait", choices=["default", "portrait", "old_photo", "game_character"], help="Pipeline preset mode.")
    parser.add_argument("--w", type=float, default=0.6, help="Fidelity weight (0.0 to 1.0).")
    parser.add_argument("--save-images", type=Path, default=None, help="Directory to save enhanced output images.")
    parser.add_argument("--output-json", type=Path, default=PROJECT_DIR / "benchmarks" / "baseline_report.json", help="Path to save output JSON metrics report.")
    args = parser.parse_args()

    if args.dry_run:
        samples = load_manifest(args.manifest)
        input_count, reference_count = validate_files(samples)
        categories = sorted({sample.category for sample in samples})
        print(f"Benchmark valid: {input_count} inputs, {reference_count} references")
        print(f"Categories: {', '.join(categories)}")
        return

    results, summary = run_evaluation(
        manifest_path=args.manifest,
        device=args.device,
        limit=args.limit,
        save_images_dir=args.save_images,
        preset=args.preset,
        fidelity_w=args.w,
    )
    
    print_markdown_report(summary)
    
    if args.output_json:
        report_data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved complete evaluation report to: {args.output_json}")


if __name__ == "__main__":
    main()
