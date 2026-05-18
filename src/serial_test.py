import serial
import time

PORT = "/dev/cu.usbmodem14302"  # replace this with your actual port
BAUD = 115200


def main():
    print(f"Opening serial port: {PORT}")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    print("Sending Happy")
    ser.write(b"H")
    time.sleep(1.5)

    print("Sending Sad")
    ser.write(b"S")
    time.sleep(1.5)

    print("Sending Angry")
    ser.write(b"A")
    time.sleep(1.5)

    print("Sending Neutral")
    ser.write(b"N")
    time.sleep(1.5)

    ser.close()
    print("Serial test complete.")


if __name__ == "__main__":
    main()
