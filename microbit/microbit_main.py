from microbit import *

# Bit:Bot XL motor pins from the project plan.
# If movement is wrong, we will adjust these pins later.
# Left motor: pin0 forward, pin8 backward
# Right motor: pin1 forward, pin12 backward

uart.init(baudrate=115200)

display.show(Image.ASLEEP)


def stop():
    pin0.write_digital(0)
    pin8.write_digital(0)
    pin1.write_digital(0)
    pin12.write_digital(0)


def forward(t=400):
    pin0.write_digital(1)
    pin1.write_digital(1)
    sleep(t)
    stop()


def backward(t=400):
    pin8.write_digital(1)
    pin12.write_digital(1)
    sleep(t)
    stop()


def spin_left(t=400):
    pin8.write_digital(1)
    pin1.write_digital(1)
    sleep(t)
    stop()


def spin_right(t=400):
    pin0.write_digital(1)
    pin12.write_digital(1)
    sleep(t)
    stop()


def wiggle():
    forward(250)
    backward(250)


stop()

while True:
    if uart.any():
        cmd = uart.read(1).decode("utf-8").strip()

        if cmd == "H":
            display.show(Image.HAPPY)
            spin_right(500)

        elif cmd == "S":
            display.show(Image.SAD)
            backward(500)

        elif cmd == "A":
            display.show(Image.ANGRY)
            backward(700)

        elif cmd == "F":
            display.show(Image.SURPRISED)
            stop()

        elif cmd == "U":
            display.show(Image.SURPRISED)
            wiggle()

        elif cmd == "D":
            display.show(Image.CONFUSED)
            backward(350)
            spin_left(350)

        elif cmd == "N":
            display.show(Image.ASLEEP)
            stop()
