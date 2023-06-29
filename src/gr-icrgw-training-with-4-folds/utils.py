#########################################
################ utils.py ###############
#########################################

import requests
import json

def send_line_notification(message, line_json_path):
    f = open(line_json_path, "r")
    json_data = json.load(f)
    line_token = json_data["kagglePush"]
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)