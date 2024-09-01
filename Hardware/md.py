import RPi.GPIO as GPIO
import time
import servo

def set_pir():
    PIR_1_PIN = 4
    PIR_2_PIN = 17

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_1_PIN , GPIO.IN)
    GPIO.setup(PIR_2_PIN , GPIO.IN)

def motion_detection(pir_pin):
    if pir_pin == PIR_1_PIN:
        print("Motion detected by PIR 1 ")
        servo.move(30, s)
    else:
        print("Motion Detected by PIR 2 ")
        servo.move(150, s)

'''
try:
    print("Start")
    time.sleep(2)

    print("Ready")
    while True:
        if GPIO.input(PIR_1_PIN):
            motion_detection(PIR_1_PIN)
        elif GPIO.input(PIR_2_PIN):
            motion_detection(PIR_2_PIN)
        time.sleep(1)

except KeyboardInterrupt:
    print("Exit")

'''