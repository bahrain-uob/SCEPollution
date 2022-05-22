import boto3 
import botocore.exceptions
import time
import math
# {'waittime': 3.4231884057971014, 'busses': 0, 'cars': 21, 'trucks': 2, 'AQI': 4, 'CO': 0.37, 'cityType': 'res', 'city': 'Madinat Hamad', 'intersectionId': '09c1dfa6-bf51-49b3-8214-f6ad11aff852', 'sensorId': 'f8735c0d-d3a6-45f0-b365-86810cbd1852'}

client = boto3.client('timestream-write')
def current_milli_time():
    return round(time.time() * 1000)

def lambda_handler(event, context):
    response = {}
    try: 
        waittime = event['waittime']
        busses = event['busses']
        cars = event['cars']
        trucks = event['trucks']
        AQI = event['AQI']
        CO = event['CO']
        cityType = event['cityType']
        city = event['city']
        intersectionId = event['intersectionId']
        sensorId = event['sensorId']

        #  insert multi-measure recrod into timestream record
        current_time = current_milli_time()
        response = client.write_records(
        DatabaseName='test',
        TableName='test',
        Records=[
            {
            'Dimensions': [
                {
                    'Name': 'cityType',
                    'Value': cityType,
                    'DimensionValueType': 'VARCHAR'
                },
                {
                    'Name': 'city',
                    'Value': city,
                    'DimensionValueType': 'VARCHAR'
                },
                {
                    'Name': 'intersectionId',
                    'Value': intersectionId,
                    'DimensionValueType': 'VARCHAR'
                },
                {
                    'Name': 'sensorId',
                    'Value': sensorId,
                    'DimensionValueType': 'VARCHAR'
                }
            ],
            'MeasureName': 'dummy_metrics',
            'MeasureValueType': 'MULTI',
            'Time': str(current_time),
            'MeasureValues': [
                {
                    'Name': 'busses',
                    'Value': str(busses),
                    'Type': 'BIGINT'
                },
                {
                    'Name': 'cars',
                    'Value': str(cars),
                    'Type': 'BIGINT'
                },
                {
                    'Name': 'trucks',
                    'Value': str(trucks),
                    'Type': 'BIGINT'
                },
                {
                    'Name': 'AQI',
                    'Value': str(AQI),
                    'Type': 'BIGINT'
                },
                {
                    'Name': 'CO',
                    'Value': str(CO),
                    'Type': 'DOUBLE'
                },
                {
                    'Name': 'waittime',
                    'Value': str(math.ceil(waittime)),
                    'Type': 'BIGINT'
                }
            ]
            }
        ]
        )
        print(response)
    except botocore.exceptions.ClientError as error:
        print('Error occurred')
        print(error)


