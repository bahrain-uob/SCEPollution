def getCountType(dictCounts, type):
    count = 0
    for k in dictCounts:
        if dictCounts[k]['class'] == type:
            count += 1
    return count


if __name__ == "__main__": 
    dic = {
        "0": {
            "class": 'car',
        },
        "1": {
            "class": 'buss'
        },
                "2": {
            "class": 'buss'
        }
        ,
                "3": {
            "class": 'buss'
        }
    }
    print(getCountType(dic, 'buss'))