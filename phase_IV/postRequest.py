# importing the requests library

import requests, time
t=time.time()
##r = requests.post('http://ssuhrid.com/test.php', params={'x1':'test','x2':789,'x3':654})
##print r.status_code
data = {'enroll':13102101,'name':'Bonga','attend':10}
r = requests.post('http://ssuhrid.com/updateDB.php',params=data)
print r.status_code
