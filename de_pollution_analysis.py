import csv
import requests
import json
import pandas as pd
import os
import zipfile,io

# Method to parse JSON
def parseJSON(content, attach, i):
  if type(content) != dict:
    resMap[i][attach] = content
  else:
    for key in content.keys():
      if type(content[key]) is list:
        for itemInList in range(len(content[key])):
          parseJSON(content[key][itemInList], attach + '@' + key + str(itemInList), i)
      elif type(content[key]) == dict:
        parseJSON(content[key], attach + '@' + key, i)
      else:
        resMap[i][attach + '@' + key] = content[key]
      

# Get Indicators from data (WHO)
url = "https://ghoapi.azureedge.net/api/Indicator"
response = requests.request("GET", url)
json_parsed = json.loads(response.text)
resMap = []
for i in range(len(json_parsed['value'])):
  resMap.append({})
  parseJSON(json_parsed['value'][i],'', i)
resDf = pd.DataFrame(resMap)
resDf.to_csv('/content/drive/My Drive/Big-Data/Project/indicators.csv')

# Get Dimensions from Data (WHO)
url = "https://ghoapi.azureedge.net/api/Dimension"
response = requests.request("GET", url)
json_parsed = json.loads(response.text)
resMap = []
for i in range(len(json_parsed['value'])):
  resMap.append({})
  parseJSON(json_parsed['value'][i],'', i)
resDf = pd.DataFrame(resMap)


# Retrieve pollution related datasets from EPA (From 1980 - 2020)
for poll in ['44201','42401', '42101', '42602', 'HAPS','VOCS', 'NONOxNOy', 'Lead', '88101','88502', '81102', 'SPEC', 'PM10SPEC']:
  t = "/content/drive/My Drive/Big-Data/Project/AP/" + poll
  os.mkdir(t)
  for year in range(1980,2020):
    t = "/content/drive/My Drive/Big-Data/Project/AP/" + poll + "/"
    url = "https://aqs.epa.gov/aqsweb/airdata/daily_" + poll + "_"+ str(year) +".zip"
    response = requests.request("GET", url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(t)

