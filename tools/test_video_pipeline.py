"""Unit test for video AI enhancement in pipeline.py."""

from __future__ import annotations

import os
import sys
import tempfile
import cv2
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from pipeline import LocalAIEnhancerPipeline


def test_video_enhancement():
    print("=== Testing Video AI Enhancement Engine ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_vid = os.path.join(tmpdir, "test_in.mp4")
        output_vid = os.path.join(tmpdir, "test_out.mp4")

        # Create a synthetic 5-frame 256x256 test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(input_vid, fourcc, 10.0, (256, 256))
        for i in range(5):
            frame = np.full((256, 256, 3), 100 + i * 20, dtype=np.uint8)
            # Draw a circle simulating a face
            cv2.circle(frame, (128, 128), 40, (220, 180, 150), -1)
            writer.write(frame)
        writer.release()

        assert os.path.exists(input_vid), "Input video failed to generate"

        pipeline = LocalAIEnhancerPipeline(device="cpu")
        
        progresses = []
        def progress_cb(stage, pct, msg):
            progresses.append((stage, pct, msg))

        stats = pipeline.process_video(
            input_video_path=input_vid,
            output_video_path=output_vid,
            w=0.5,
            detection_model="retinaface_mobile0.25",
            upscale=1,
            frame_stride=1,
            max_frames=5,
            progress_callback=progress_cb
        )

        assert os.path.exists(output_vid), "Output video was not created"
        assert stats["total_frames"] == 5, f"Expected 5 frames, got {stats['total_frames']}"
        assert len(progresses) > 0, "No progress callbacks received"
        print(f"[OK] Video processed successfully: {stats}")

    print("SUCCESS: Video AI Enhancement pipeline verified cleanly with exit code 0!")


if __name__ == "__main__":
    test_video_enhancement()
