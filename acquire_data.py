import requests
import json
import time
import os
from datetime import datetime
import cv2

headers = {
    "Content-Type": "application/json"
}

def login(camera_data, username, password):
    url = f'http://{camera_data}/api.cgi?cmd=Login'
    data = [{
        "cmd": "Login",
        "param": {
            "User": {
                "Version": "0",
                "userName": username,
                "password": password
            }
        }
    }]
    
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    result = response.json()
    TOKEN = result[0]["value"]["Token"]["name"]
    if response.status_code == 200: 
        print(f" --> IP CAM Login Successful")
    return TOKEN

def set_white_led(camera_data, token, state):
    url = f'http://{camera_data}/api.cgi?cmd=SetWhiteLed&token={token}'
    data = [{
        "cmd": "SetWhiteLed",
        "param": {
            "WhiteLed": {
                "channel": 0,
                "state": state
            }
        }
    }]
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    if response.status_code == 200 and state == 0: 
        print(f" --> Turned off the IR LED")
    elif response.status_code == 200 and state == 1: 
        print(f" --> Turned on the IR LED")
    else: 
        print(f" --> IR LED Error")

def get_snap(camera_data, token):
    url = f'http://{camera_data}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=flsYJfZgM6RTB_os&token={token}'
    response = requests.post(url=url, headers=headers)
    if response.status_code == 200: 
        print(f" --> Snapshot Successful")
    return response.content 

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f" --> Created folder: {output_folder}")

def save_frame(frame, save_dir):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f'{timestamp_str}.jpg'
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(frame)
    return filepath

def logout(camera_data, token):
    url = f'http://{camera_data}/api.cgi?cmd=Logout&token={token}'
    data = [{
        "cmd": "Logout",
        "param": {}
    }]
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    if response.status_code == 200: 
        print(f" --> Logout Succesful")

def render(image, string):
    cv2.imshow(string, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def acquire_data():
    with open('/home/asiadmin/Workspace/CENTRAL_FINAL/config_ipcam.json', 'r') as f:
        config = json.load(f)
        camera_data = config['camera_data']
        username = config['username']
        password = config['password']
    results = [{} for _ in range(len(camera_data))] 
    save_dir = "/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/IMAGES/"
    create_output_folder(save_dir)
    for i in range(len(camera_data)):
        curr_cam = camera_data[i]['name']
        if camera_data[i]['status'] == 0:
            curr_subdirectory = os.path.join(save_dir, curr_cam)
            create_output_folder(curr_subdirectory)
            curr_ip = camera_data[i]['ip']
            token = login(curr_ip, username, password)
            set_white_led(curr_ip, token, 1)
            camera_data[i]['status'] = 1
            time.sleep(2)
            snapshot = get_snap(curr_ip, token)
            saved_image = save_frame(snapshot, curr_subdirectory)
            set_white_led(camera_data=curr_ip, token=token, state=0) 
            camera_data[i]['status'] = 0
            logout(curr_ip, token)
            results[i].update({"curr_cam" : curr_cam, "acquisition_path" : saved_image})
    return results 

