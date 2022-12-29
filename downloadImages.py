# -*- coding: utf-8 -*-
"""imgs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m03gz2kFDzxjdlT1kn1uaUrPBhFcGnvS
"""

import json

with open('./export.json') as f:
  data = json.load(f)

import urllib.request

def saveImages(json):
    for data in json:
        orgImgName = data['External ID']
        try:
            if data['Label']['objects'][0].get('instanceURI'):
                if len(data['Reviews']) < 1 or (len(data['Reviews']) > 0 and data[0]['Reviews'][0]["score"] != -1):
                    maskImgName = data['External ID'].split(".")[0] + '.PNG'
                    urllib.request.urlretrieve(data['Labeled Data'], 'Original/'+orgImgName)
                    urllib.request.urlretrieve(data['Label']['objects'][0]['instanceURI'], 'Mask/'+orgImgName)
                    urllib.request.urlretrieve(data['Label']['objects'][0]['instanceURI'], 'Mask_PNG/'+maskImgName)
        except KeyError:
            print(orgImgName + " not applicable")

saveImages(data)
