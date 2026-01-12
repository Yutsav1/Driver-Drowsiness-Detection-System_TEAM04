import argparse
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import pygame

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# ================= AUDIO SETUP =================
pygame.mixer.init()

AUDIO_MAP = {
    "awake": "audio/awake.wav",
    "sleeping": "audio/sleeping.wav",
    "sunglass_detected": "audio/sunglass_detected.wav"
}

last_audio_state = None
last_detections = []
last_results = None


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
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def draw_detections(request, stream="main"):
    detections = last_results
    if not detections:
        return

    labels = get_labels()

    with MappedArray(request, stream) as m:
        frame = m.array
        detected_state = None

        for det in detections:
            x, y, w, h = det.box
            label = labels[int(det.category)]
            conf = det.conf

            if label == "awake":
                box_color = (0, 255, 0)
                detected_state = "awake"
                status_text = "Driver Alert - Safe Driving"
                text_color = (0, 255, 0)

            elif label == "sleeping":
                box_color = (0, 0, 255)
                detected_state = "sleeping"
                status_text = "Driver Not Alert"
                text_color = (0, 0, 255)

            elif label == "sunglass_detected":
                box_color = (0, 255, 0)
                detected_state = "sunglass_detected"
                status_text = "Sunglasses Detected"
                text_color = (0, 255, 255)

            else:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        if detected_state:
            draw_status_banner(frame, status_text, text_color)
            play_audio_once(detected_state)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="xy")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


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
