import os
import numpy as np
import onnxruntime as ort


MODEL_PATH = "models/emotion_model.onnx"


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name

    print("Model loaded successfully.")
    print("Input name:", input_name)
    print("Input shape:", sess.get_inputs()[0].shape)
    print("Output shape:", sess.get_outputs()[0].shape)

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = sess.run(None, {input_name: dummy})[0]

    print("Dummy output shape:", output.shape)
    print("Dummy prediction index:", output.argmax(axis=1)[0])


if __name__ == "__main__":
    main()
