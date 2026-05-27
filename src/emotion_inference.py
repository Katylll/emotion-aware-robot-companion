import os
import time

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import onnxruntime as ort

import serial
from collections import deque

MODEL_PATH = "models/emotion_model.onnx"
SERIAL_PORT = "/dev/cu.usbmodem14302"  # replace with your actual micro:bit port
BAUD_RATE = 115200
ENABLE_ROBOT = True

CMD_MAP = {
    "Angry": b"A",
    "Disgust": b"D",
    "Fear": b"F",
    "Happy": b"H",
    "Neutral": b"N",
    "Sad": b"S",
    "Surprise": b"U",
}

# IMPORTANT:
# This order must match PyTorch ImageFolder alphabetical class order.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()


def preprocess_face(face_crop):
    """
    Convert OpenCV BGR face crop into MobileNetV2 input:
    BGR -> RGB, resize 224x224, normalize, CHW, batch dimension.
    """
    img = cv2.resize(face_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # CHW -> NCHW

    return img.astype(np.float32)


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    print("Loaded ONNX model:", model_path)
    print("Input name:", input_name)

    return sess, input_name


def infer_emotion(face_crop, sess, input_name):
    inp = preprocess_face(face_crop)
    logits = sess.run(None, {input_name: inp})[0][0]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    emotion = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])

    return emotion, confidence


def create_oak_camera_queue():
    """
    DepthAI v3 camera pipeline.
    Uses Camera node and direct output queue, matching your working Stage 1 camera test.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build()
    preview = cam.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.RGB888p
    )

    q = preview.createOutputQueue()

    pipeline.start()

    return pipeline, q


def main():
    sess, input_name = load_onnx_model(MODEL_PATH)
    ser = None

    if ENABLE_ROBOT:
        print(f"Opening robot serial port: {SERIAL_PORT}")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Robot serial connected.")
   
    history = deque(maxlen=7)
    last_sent = None
    last_send_time = 0
    SEND_COOLDOWN = 1.5
    mp_face_detection = mp.solutions.face_detection

    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

    pipeline, q = create_oak_camera_queue()

    last_time = time.time()
    fps = 0.0

    with pipeline:
        while pipeline.isRunning():
            frame = q.get().getCvFrame()

            # MediaPipe expects RGB input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            display_label = "No face"

            if results.detections:
                # Use the first detected face
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box

                h, w = frame.shape[:2]

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)

                # Add small margin around face
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    emotion, conf = infer_emotion(face_crop, sess, input_name)

                    if conf < 0.50:
                        display_label = f"Uncertain {conf:.0%}"
                        stable_emotion = None
                    else:
                        display_label = f"{emotion} {conf:.0%}"
                        stable_emotion = emotion

                    if ENABLE_ROBOT and ser is not None and stable_emotion is not None:
                        history.append(stable_emotion)

                        now = time.time()
                        is_stable = len(history) == history.maxlen and len(set(history)) == 1
                        cooldown_passed = now - last_send_time > SEND_COOLDOWN

                        if is_stable and stable_emotion != last_sent and cooldown_passed:
                            cmd = CMD_MAP.get(stable_emotion)

                            if cmd is not None:
                                ser.write(cmd)
                                print(f"Sent to robot: {stable_emotion} -> {cmd}")

                                last_sent = stable_emotion
                                last_send_time = now
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
                    cv2.putText(
                        frame,
                        display_label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 100),
                        2
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
                    cv2.putText(
                        frame,
                        display_label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 100),
                        2
                    )

            # FPS calculation
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now

            cv2.putText(
                frame,
                f"Status: {display_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Emotion Robot - Stage 4", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    if ser is not None:
        ser.write(b"N")
        time.sleep(0.2)
        ser.close()
        print("Robot serial closed.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
