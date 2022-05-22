import sys
import pymongo 
import os
from datetime import datetime
import time
##Create a MongoDB client, open a connection to Amazon DocumentDB as a replica set and specify the read preference as secondary preferred

def lambda_handler(event, context):
    client = pymongo.MongoClient('mongodb://<user-name>:<password>@docdb-2022-05-22-12-10-20.cluster-c5hpt65t5qqx.us-east-1.docdb.amazonaws.com:27017/?ssl=true&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false') 
    # sensorId = event['sensorId']
    # ##Specify the database to be used
    db = client.sample_database
    
    # insert intersectioncollection 
    # db.intersectioncollection.insertOne({ "intersectionID" : "2d5299f0-db28-4562-b1b5-011a02843c8b", "location" : { "coordinates" : [ 50.58213, 26.232337 ], "type" : "Point" }, "cityID" : "0705bc80-e847-4a93-8220-a346719b12b5", "sensors" : [ { "sensorID" : "77675157-46aa-45f7-b12e-62f3bcd1186b", "lastPingTime" : ISODate() } ] })
    # ##Specify the collection to be used
    col = db.intersectioncollection
    
    my_date = datetime.now()
    print(my_date)
    col.insert_one({"intersectionID" : "2d5299f0-db28-4562-b1b5-011a02843c8b", "location" : { "coordinates" : [ 50.58213, 26.232337 ], "type" : "Point" }, "cityID" : "0705bc80-e847-4a93-8220-a346719b12b5", "sensors" : [ { "sensorID" : "77675157-46aa-45f7-b12e-62f3bcd1186b", "lastPingTime" : my_date.isoformat()} ] })
    x = col.find_one({'intersectionID':"2d5299f0-db28-4562-b1b5-011a02843c8b"})
    print(x)

    #  wait for 5 seconds
    time.sleep(5)
    my_date = datetime.now()
    filter = { 'sensors.sensorID': "77675157-46aa-45f7-b12e-62f3bcd1186b" }
    col.update_one(filter,{"$set":{'sensors.$.lastPingTime': my_date.isoformat()}})
    # ##Insert a single document
    # ##Find the document that was previously written
    x = col.find_one({'intersectionID':'2d5299f0-db28-4562-b1b5-011a02843c8b'})
    # ##Print the result to the screen
    print(x)

    # ##Close the connection
    client.close()