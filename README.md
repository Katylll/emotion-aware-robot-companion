# Emotion-Aware Robot Companion

A robot machine learning project that recognises facial expression categories in real time and maps them to physical Bit:Bot XL behaviours.

The system uses:

* **OAK-D S2** for live video capture
* **MediaPipe** for face detection
* **MobileNetV2** trained on FER2013 for expression classification
* **ONNX Runtime** for local real-time inference
* **BBC micro:bit and Bit:Bot XL** for robot movement and LED responses

> The system predicts visible facial expression categories, not a person’s internal emotional state.

## Project Pipeline

```text
Camera frame
→ Face detection
→ Image preprocessing
→ MobileNetV2 inference
→ Confidence and smoothing checks
→ Serial command
→ Bit:Bot robot response
```

## Project Structure

```text
emotion_robot/
├── notebooks/     # Dataset analysis, training and evaluation
├── src/           # Camera, inference and serial-control scripts
├── microbit/      # MicroPython code for the micro:bit
├── models/        # Trained PyTorch and ONNX model files
├── figures/       # Charts, confusion matrix and evaluation results
├── data/          # Raw and processed FER2013 data
├── reports/       # Assignment reports
├── requirements.txt
└── README.md
```

## Important Files

* `notebooks/01_fer2013_eda_preprocessing.ipynb` — dataset analysis and preprocessing
* `notebooks/02_mobilenet_training_evaluation.ipynb` — MobileNetV2 training and evaluation
* `notebooks/03_svm_baseline.ipynb` — SVM baseline comparison
* `src/camera_test.py` — tests the OAK-D S2 camera
* `src/serial_test.py` — tests Python-to-micro:bit communication
* `src/test_onnx_model.py` — checks that the ONNX model loads correctly
* `src/emotion_inference.py` — runs live facial-expression inference and robot control
* `src/robot_serial_test.py` — sends test commands directly to the robot
* `microbit/microbit_main.py` — receives serial commands and controls the Bit:Bot

## Setup

Create and activate the Conda environment:

```bash
conda create -n emotion-robot python=3.10
conda activate emotion-robot
pip install -r requirements.txt
```

The following model files must be present locally:

```text
models/emotion_model.onnx
models/emotion_model.onnx.data
```

## Run Live Inference Only

In `src/emotion_inference.py`, set:

```python
ENABLE_ROBOT = False
```

Run from the project root:

```bash
conda activate emotion-robot
python src/emotion_inference.py
```

The camera window displays the face box, predicted expression, confidence and FPS. Click the camera window and press `q` to close it.

## Run the Full Robot Demonstration

1. Flash `microbit/microbit_main.py` to the micro:bit.
2. Insert the micro:bit into the Bit:Bot XL.
3. Connect both the OAK-D S2 and micro:bit to the computer.
4. Turn on the Bit:Bot battery power.
5. Find the micro:bit serial port:

```bash
ls /dev/cu.*
```

6. Update these settings in `src/emotion_inference.py`:

```python
ENABLE_ROBOT = True
SERIAL_PORT = "/dev/cu.usbmodemXXXX"
BAUD_RATE = 115200
```

7. Run:

```bash
python src/emotion_inference.py
```

The program performs live inference and sends stable predictions to the robot after confidence, smoothing and cooldown checks.

## Model Results

* **MobileNetV2 validation accuracy:** 66.98%
* **SVM baseline accuracy:** 28.31%
* Strongest classes: Happy and Surprise
* Weaker classes: Fear and Sad

## Notes

The correct model class order is:

```python
["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
```

Keep `emotion_model.onnx` and `emotion_model.onnx.data` in the same folder.

Close the micro:bit browser editor before running Python because it may occupy the serial port.

Large dataset and model files are kept locally and may be excluded from GitHub.

