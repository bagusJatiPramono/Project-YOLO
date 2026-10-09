"""
Vehicle Tracking & Counting with YOLOv8
-----------------------------------------
- Detects only within a defined ROI box (center=[744,540], w=640, h=225)
- Tracks vehicles across frames (ByteTrack, via Ultralytics .track())
- Counts a vehicle once when its center crosses the counting line
- Saves results to an Excel file: counted_data.xlsx

Usage:
    python vehicle_counter.py --source path/to/video.mp4
    python vehicle_counter.py --source path/to/video.mp4 --show --save-video
"""

import argparse
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# ROI CONFIG
# ---------------------------------------------------------------------------
ROI_CENTER = (744, 540)
ROI_WIDTH = 640
ROI_HEIGHT = 225

ROI_X1 = ROI_CENTER[0] - ROI_WIDTH // 2
ROI_X2 = ROI_CENTER[0] + ROI_WIDTH // 2
ROI_Y1 = ROI_CENTER[1] - ROI_HEIGHT // 2
ROI_Y2 = ROI_CENTER[1] + ROI_HEIGHT // 2

# Counting line: vertical line through the middle of the ROI
LINE_Y = ROI_CENTER[1]
LINE_X1 = ROI_X1
LINE_X2 = ROI_X2

# COCO vehicle classes: car, motorcycle, bus, truck
VEHICLE_CLASSES = {0: "motorcycle", 1: "car"}


def point_in_roi(cx, cy):
    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


def main(args):
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (frame_w, frame_h))

    # track_id -> last known y-center (to detect line crossing)
    prev_y = {}
    # track_id -> already counted (True/False)
    counted_ids = set()

    counted_records = []
    frame_idx = 0

    # persist=True keeps the tracker state across frames (needed for stable IDs)
    for result in model.track(
        source=args.source,
        stream=True,
        persist=True,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker="bytetrack.yaml",
        conf=args.conf,
        verbose=False,
    ):
        frame = result.orig_img
        frame_idx += 1
        timestamp_sec = frame_idx / fps

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, tid, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Only consider detections whose center is inside the ROI
                if not point_in_roi(cx, cy):
                    continue

                class_name = VEHICLE_CLASSES.get(cls_id, str(cls_id))

                # Draw box + id for visualization
                if args.show or args.save_video:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"ID:{tid} {class_name}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

                # Line-crossing check: did the track move from one side of
                # LINE_Y to the other between this frame and the last?
                if tid in prev_y and tid not in counted_ids:
                    prev_cy = prev_y[tid]
                    crossed_right = prev_cy < LINE_Y <= cy
                    crossed_left = prev_cy > LINE_Y >= cy

                    if crossed_right or crossed_left:
                        direction = "right" if crossed_right else "left"
                        counted_ids.add(tid)
                        counted_records.append({
                            "track_id": int(tid),
                            "vehicle_type": class_name,
                            "direction": direction,
                            "confidence": round(float(conf), 3),
                            "frame": frame_idx,
                            "time_sec": round(timestamp_sec, 2),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        print(f"[COUNTED] id={tid} type={class_name} dir={direction} "
                              f"frame={frame_idx}")

                prev_y[tid] = cy

        if args.show or args.save_video:
            # Draw ROI box
            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
            # Draw counting line
            cv2.line(frame, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (0, 0, 255), 2)
            # Running count
            cv2.putText(frame, f"Count: {len(counted_ids)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if args.save_video and writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Vehicle Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ---------------------------------------------------------------------
    # Save results to Excel
    # ---------------------------------------------------------------------
    df = pd.DataFrame(counted_records)
    output_path = "counted_data.xlsx"

    if not df.empty:
        # Add a summary sheet alongside the detailed log
        summary = df.groupby(["vehicle_type", "direction"]).size().reset_index(name="count")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer_xl:
            df.to_excel(writer_xl, sheet_name="counted_data", index=False)
            summary.to_excel(writer_xl, sheet_name="summary", index=False)
    else:
        df = pd.DataFrame(columns=[
            "track_id", "vehicle_type", "direction", "confidence",
            "frame", "time_sec", "timestamp",
        ])
        df.to_excel(output_path, sheet_name="counted_data", index=False)

    print(f"\nTotal vehicles counted: {len(counted_ids)}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle tracking & counting")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    args = parser.parse_args()

    main(args)