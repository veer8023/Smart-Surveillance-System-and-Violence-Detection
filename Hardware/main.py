import RPi.GPIO as GPIO
import time

PIR_1_PIN = 4
PIR_2_PIN = 17

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(36,GPIO.OUT)
    s = GPIO.PWM(36,50)
    s.start(0)
    return s

def terminate(s):
    s.stop()
    GPIO.cleanup()

def move(angle):
    s = setup()
    duty_cycle = 3 + 9*(angle/180)
    s.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    terminate(s)

def set_pir():    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_1_PIN , GPIO.IN)
    GPIO.setup(PIR_2_PIN , GPIO.IN)

def motion_detection(pir_pin):
    if pir_pin == PIR_1_PIN:
        print("Motion detected by PIR 1 ")
        move(150)
    else:
        print("Motion Detected by PIR 2 ")
        move(30)

try:
    
    print("Start")
    time.sleep(2)

    print("Ready")
    
    while True:
        set_pir()
        pir1 = GPIO.input(PIR_1_PIN)
        pir2 = GPIO.input(PIR_2_PIN)
        GPIO.cleanup()
        if pir1 and pir2:
            move(90)
        elif pir1:
            motion_detection(PIR_1_PIN)
        elif pir2:
            motion_detection(PIR_2_PIN)
        else:
            time.sleep(1)
        
except KeyboardInterrupt:
    print("Exit")