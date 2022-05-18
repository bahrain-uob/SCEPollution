from readingSerial1 import COreader 
import threading 

if __name__ == '__main__': 
    coreader_object = COreader(arduino_port='COM3') 
    t = threading.Thread(target=coreader_object.main, daemon= True)
    t.start() 

    while True:
        print("main:")
        avg_co,avg_aqi = coreader_object.q.get()
        print(avg_co)
        print(avg_aqi)