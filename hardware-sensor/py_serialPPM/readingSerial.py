# Importing Libraries
import serial
import time

# Set arduino serial COM3
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

# Function to read from the arduino
def write_read():
    data = arduino.readline()
    # print(data)
    return data

# AQI Function to calculate the Air Quality
def gen_aqi(co):
    if co<=4.4:
        Ih=50
        Il=0
        BPh=4.4
        BPl=0
    elif co<=9.4:
        Ih=100
        Il=51
        BPh=9.4
        BPl=4.5
    elif co<=12.4:
        Ih=150
        Il=101
        BPh=12.4
        BPl=9.5
    elif co<=15.4:
        Ih=200
        Il=151
        BPh=15.4
        BPl=12.5
    elif co<=30.4:
        Ih=300
        Il=201
        BPh=30.4
        BPl=15.5
    elif co<=40.4:
        Ih=400
        Il=301
        BPh=40.4
        BPl=30.5
    else:
        Ih=500
        Il=401
        BPh=50.4
        BPl=40.5
    aqi=int(((Ih-Il)/(BPh-BPl))*(co-BPl)+Il)
    return aqi

# Declaring variables for the calculation
count = 0
start = time.time()
stop = time.time()
sum = 0.0
avg = 0.0
Zeros = False
AQI = ''

# Infinite loop for the continuous reading and calculation
while True:
    # Save the arduino read value and convert from Byte to String
    value = write_read().decode('UTF-8')
    # if the value null consider it as Zero
    if value == '':
        value = "0"
        Zeros = True 
    else:
        Zeros = False
    # Extract the digit values into one String 
    emp_str = ""
    for m in value:
        if m.isdigit():
            emp_str = emp_str + m
    
    # Clear the string value from newline character
    valueclear = emp_str.strip('\n')
    valueclear = valueclear.strip('\r')
    valueclear = valueclear.strip('')
    valueclear = valueclear.rstrip('\r')
    valueclear = valueclear.rstrip('\n')
    valueclear = valueclear.rstrip('')

    # Convert string to float and divide it by 100
    valuenum = int(float(valueclear.rstrip()))/100

    # Check the value if it is Zero or non-Zero
    if Zeros == False:

        # Checking if the time didn't reached the defined period
        if (stop - start) < 3600:

            # Get the sum, update the end-time, increament the count
            sum = sum + valuenum
            stop = time.time()
            count = count + 1
        else:
            # if the time reached the period, get the avarage and update start-time and end-time
            avg = sum / count
            start = time.time()
            stop = start
        print("Time: ", stop - start)
        if avg == 0.0:
            continue
        else:
            print("Avg CO: ", avg) # printing the value
            print("AQI Avg: ", gen_aqi(avg))
            avg = 0.0
            sum = 0.0
            count = 0
    else:
        # Update end-time to skip the Zero values
        stop = time.time()

