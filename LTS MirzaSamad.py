import csv
import requests
import os, sys

from bs4 import BeautifulSoup
from datetime import datetime
import json
from httplib2 import FileCache


#reload(sys)
#sys.setdefaultencoding('UTF8')

html = """ <resources><string name="app_name">Nirdesh app</string><string name=" for moving">Open navigation drawer</string><string name="moving down button">Close navigation drawer</string><string name="you are inside app">Settings</string>"""
# you can put any text that you want to convert 

try:
    
    soup=BeautifulSoup(html)
    print(soup.find_all('string'))
    url='https://translation.googleapis.com/language/translate/v2/?key=AIzaSyD97Bja9raG9-LKJtVDpVtgv4bS0TdD2Tw'
    for string in soup.find_all('string'):
        print(string.text)
        payload = {
                "q": ""+string.text,
                "target": "hi",
                "source": "en",
                "format": "text"
            }
        headers = {'content-type': 'application/json'}
        try:
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            #print(response)
            jsonValue= json.loads(response.text)
            rowValue=""
            print(jsonValue['data']['translations'][0]['translatedText'])
            #string.text=str(jsonValue['data']['translations'][0]['translatedText'])
            tag=string
            tag.string=jsonValue['data']['translations'][0]['translatedText']
            print(string)
            rowValue=""+str(string)
            f = open('helloworld9.txt','a')
            f.write('\n' + rowValue)
            f.close()
        except Exception as R:
            print(R)
            print("error in api hit")
    #print(soup)


except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    # import pdb;pdb.set_trace()
    print(e)
    print("this is exception part")
