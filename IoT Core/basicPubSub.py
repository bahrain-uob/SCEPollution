'''
/*
 * Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
 '''

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json
import serial 

# Set arduino serial COM3
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

# Function to read from the arduino
def write_read():
    data = arduino.readline()
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
Msg = ''

AllowedActions = ['both', 'publish', 'subscribe']

# Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")


# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                    help="Targeted client id")
parser.add_argument("-t", "--topic", action="store", dest="topic", default="sdk/test/Python", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")

args = parser.parse_args()
host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
port = args.port
useWebsocket = args.useWebsocket
clientId = args.clientId
topic = args.topic

if args.mode not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
    exit(2)

if args.useWebsocket and args.certificatePath and args.privateKeyPath:
    parser.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
    parser.error("Missing credentials for authentication.")
    exit(2)

# Port defaults
if args.useWebsocket and not args.port:  # When no port override for WebSocket, default to 443
    port = 443
if not args.useWebsocket and not args.port:  # When no port override for non-WebSocket, default to 8883
    port = 8883

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec

# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()
if args.mode == 'both' or args.mode == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
time.sleep(2)

# Publish to the same topic in a loop forever
loopCount = 0
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
        if (stop - start) < 20:

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
            Msg = ("Avg CO: ", avg, "\nAQI Avg: ", gen_aqi(avg))
            avg = 0.0
            sum = 0.0
            count = 0
    else:

        # Update end-time to skip the Zero values
        stop = time.time()
    
    if (stop - start) >= 20:
        if args.mode == 'both' or args.mode == 'publish':
            message = {}
            message['messages'] = Msg
            message['sequence'] = loopCount
            messageJson = json.dumps(message)
            myAWSIoTMQTTClient.publish(topic, messageJson, 1)
            if args.mode == 'publish':
                print('Published topic %s: %s\n' % (topic, messageJson))

            loopCount += 1
        time.sleep(1)
