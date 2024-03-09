import requests
r = requests.get("https://dlgr-b27bf974.herokuapp.com/recruitbutton/1")
print(r.status_code)