import RPi.GPIO as GPIO
import os
import time

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(36,GPIO.OUT)
    s = GPIO.PWM(36,50)
    s.start(0)
    return s

def move(angle, s):
    duty_cycle = 3 + 9*(angle/180)
    s.ChangeDutyCycle(duty_cycle)
    time.sleep(2)
    
def terminate(s):
    s.stop()
    GPIO.cleanup()
    
'''

s = setup()
move(30, s)
move(150, s)
terminate(s)

'''