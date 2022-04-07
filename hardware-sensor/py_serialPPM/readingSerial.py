# Importing Libraries
import serial
import time
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)
def write_read():
    data = arduino.readline()
    return data
while True:
    value = write_read()
    print(value) # printing the value
