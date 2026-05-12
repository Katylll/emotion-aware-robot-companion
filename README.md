# Emotion-Aware Robot Companion

This project builds an emotion-aware robot companion using facial expression recognition and robot behaviour control.

## Project Concept

The system detects a person's facial expression using an OAK-D S2 camera, classifies the expression using a machine learning model, and sends commands to a BBC micro:bit / Bit:Bot XL robot to trigger physical responses.

## Hardware

- OAK-D S2 camera
- BBC micro:bit
- Bit:Bot XL
- Laptop running Python

## Software

- Python 3.10
- DepthAI
- OpenCV
- PySerial
- PyTorch
- ONNX Runtime
- MediaPipe

## Project Structure

```text
emotion_robot/
├── notebooks/      # EDA and model training notebooks
├── src/            # Python scripts for camera, inference, and robot control
├── microbit/       # MicroPython code for micro:bit
├── data/           # Dataset files, ignored by GitHub
├── models/         # Trained model files, ignored by GitHub
├── figures/        # Visualisations for reports
└── reports/        # Report drafts and PDFs
