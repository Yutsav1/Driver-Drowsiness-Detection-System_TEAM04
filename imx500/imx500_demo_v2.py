import argparse
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import pygame

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# ================= AUDIO SETUP =================
pygame.mixer.init()

AUDIO_MAP = {
    "awake": "audio/awake.wav",
    "sleeping": "audio/sleeping.wav",
    "sunglass_detected": "audio/sunglass_detected.wav",
    "danger": "audio/danger.wav"
}

last_audio_state = None
last_detections = []
last_results = None

# ================= TIMER STATE =================
sleeping_start_time = None
danger_played = False


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def play_audio_once(state):
    global last_audio_state
    if state != last_audio_state and state in AUDIO_MAP:
        pygame.mixer.music.load(AUDIO_MAP[state])
        pygame.mixer.music.play()
        last_audio_state = state


def parse_detections(metadata):
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    input_w, input_h = imx500.get_input_size()

    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

    if intrinsics.bbox_normalization:
        boxes = boxes / input_h

    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]

    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > args.threshold
    ]

    return last_detections


@lru_cache
def get_labels():
    return intrinsics.labels


def draw_status_banner(frame, text, color):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def draw_detections(request, stream="main"):
    global sleeping_start_time, danger_played

    detections = last_results
    if not detections:
        sleeping_start_time = None
        danger_played = False
        return

    labels = get_labels()

    priority = {
        "sleeping": 3,
        "sunglass_detected": 2,
        "awake": 1
    }

    best_state = None
    best_prio = 0

    with MappedArray(request, stream) as m:
        frame = m.array

        for det in detections:
            x, y, w, h = det.box
            label = labels[int(det.category)]
            conf = det.conf

            if label not in priority:
                continue

            if label == "sleeping":
                box_color = (0, 0, 255)
            elif label == "awake":
                box_color = (0, 255, 0)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if priority[label] > best_prio:
                best_prio = priority[label]
                best_state = label

        now = time.monotonic()

        # ================= SLEEPING LOGIC =================
        if best_state == "sleeping":
            if sleeping_start_time is None:
                sleeping_start_time = now
                danger_played = False

            elapsed = now - sleeping_start_time

            if elapsed >= 5.0:
                draw_status_banner(
                    frame,
                    f"DANGER! WAKE UP!  Sleeping: {elapsed:.1f}s",
                    (0, 0, 255)
                )
                if not danger_played:
                    play_audio_once("danger")
                    danger_played = True
            else:
                draw_status_banner(
                    frame,
                    f"Driver Not Alert  Sleeping: {elapsed:.1f}s",
                    (0, 0, 255)
                )
                play_audio_once("sleeping")

        else:
            sleeping_start_time = None
            danger_played = False

            if best_state == "awake":
                draw_status_banner(
                    frame,
                    "Driver Alert - Safe Driving",
                    (0, 255, 0)
                )
                play_audio_once("awake")

            elif best_state == "sunglass_detected":
                draw_status_banner(
                    frame,
                    "Sunglasses Detected",
                    (0, 255, 255)
                )
                play_audio_once("sunglass_detected")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="xy")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


# ================= MAIN =================
if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"

    with open(args.labels, "r") as f:
        intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(buffer_count=12)

    picam2.start(config, show_preview=True)

    last_results = None
    picam2.pre_callback = draw_detections

    while True:
        last_results = parse_detections(picam2.capture_metadata())
