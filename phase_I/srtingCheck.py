import PyTesser.pytesser as pyt
import Image
import requests
import cv2
import numpy as np

string = 'JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY\nA-10,SECTOR-62, NOIDA (Tel.: 0120-2400973/ 4}\nB.T-ECE-2013   9\n ; i\n` `   ABHISHEK KISHORE KAUL    \nga T\': Enroll. No.:15102030 4*  \n5%   Address; C-101, Aashirwaad Aangan, lndra vihar,\n    Kota (Rajasthan) - 324005\nTele No.: 7838097163l `-% ii ,W i,.;,\n  I 9829038901 I 4\'^ \~ 7 ~\nValid upto: JUNE 2017 Registrar\n\n'
##string = 'JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY\nA-10,SECTOR-62, NOIDA (Tel.: 0120-2400973/ 4}\nB.T-ECE-2013   9\n ; i\n` `   PRIYANK LALPURIA    \nga T\': Enroll. No.:15102030 4*  \n5%   Address; C-101, Aashirwaad Aangan, lndra vihar,\n    Kota (Rajasthan) - 324005\nTele No.: 7838097163l `-% ii ,W i,.;,\n  I 9829038901 I 4\'^ \~ 7 ~\nValid upto: JUNE 2017 Registrar\n\n'


cam=cv2.VideoCapture(1);
while True:
    ret,img=cam.read();
    cv2.imshow('image',img);
    if 0xFF & cv2.waitKey(1) == 27:
        break

string = pyt.image_to_string(Image.fromarray(np.array(img, dtype='uint8')), cleanup=True)
print string

senList = string.split('\n')
enrol = ''
name = ''
for i,sen in enumerate(senList):
    wordList = sen.split(' ')
    for j,word in enumerate(wordList):
        if 'Enro' in word:
            enroll = wordList[j+1][-8:]
            name = senList[i-1]

response = requests.get('http://ssuhrid.com/checkID.php')

res= str(response.content)
response = res.split('<tr>')[1:]

enrollList = []
nameList = []
flag = False
for i,row in enumerate(response):
    cols = row.split('<td>')
    enrollList.append(cols[1][:-5])
    nameList.append(cols[2][:-5].upper())

for i in range(0,len(enroll)):
##    print enrollList[i],enroll
##    print nameList[i],name
    if enrollList[i] in enroll and nameList[i] in name:
        flag = True
if flag == True:
    print 'ID Card is valid'
else:
    print 'ID Card not valid'
        
