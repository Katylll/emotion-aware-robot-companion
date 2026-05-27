import time
import serial


SERIAL_PORT = "/dev/cu.usbmodem14302" 
BAUD_RATE = 115200


COMMANDS = {
    "Happy": b"H",
    "Sad": b"S",
    "Angry": b"A",
    "Fear": b"F",
    "Surprise": b"U",
    "Disgust": b"D",
    "Neutral": b"N",
}


def main():
    print(f"Opening serial port: {SERIAL_PORT}")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Give micro:bit time to reset after serial opens
    time.sleep(2)

    for emotion, cmd in COMMANDS.items():
        print(f"Sending {emotion}: {cmd}")
        ser.write(cmd)
        time.sleep(1.5)

    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()
