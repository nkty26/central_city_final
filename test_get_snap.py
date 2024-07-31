import requests
import json
import time
import os 
from datetime import datetime, timedelta
import cv2 
# Headers
headers = {
    "Content-Type": "application/json"
}

def login(camera_ip, username, password):
    url = f'http://{camera_ip}/api.cgi?cmd=Login'
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
    # print(response)
    print(response.text)
    
    result = response.json()
    TOKEN = result[0]["value"]["Token"]["name"]
    print(TOKEN)
    return TOKEN

def set_white_led(camera_ip, token, state):
    url = f'http://{camera_ip}/api.cgi?cmd=SetWhiteLed&token={token}'
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
    print(response)
    print(response.text)

def get_snap(camera_ip, token):
    url = f'http://{camera_ip}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=flsYJfZgM6RTB_os&token={token}'
    response = requests.post(url=url, headers=headers)
    print(response)
    return response.content 

def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

def save_frame(frame, save_dir):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f'{timestamp_str}.jpg'
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
      f.write(frame)

def logout(camera_ip, token):
    url = f'http://{camera_ip}/api.cgi?cmd=Logout&token={token}'
    data = [{
        "cmd": "Logout",
        "param": {}
    }]
    
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(response)
    print(response.text)
def render(image,string):
    cv2.imshow(string, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    with open('/home/asiadmin/Workspace/test_ipcam_led_onoff/config_ipcam.json', 'r') as f:
        config = json.load(f)
        camera_ip = config['camera_ips']
        username = config['username']
        password = config['password']
    
    # 나중에 카메라 추가시 camera_ip[i]['status'] = 0 로 일괄적용 
    
    save_dir = "/home/asiadmin/Workspace/test_ipcam_led_onoff/IMAGES"
    create_output_folder(save_dir) # 상위 디렉토리 생성 
    for i in range(len(camera_ip)):
        if camera_ip[i]['status'] == 0: # 현재 가동여부 OFF 
            curr_subdirectory = "/home/asiadmin/Workspace/test_ipcam_led_onoff/IMAGES/" + "CAM_" + str(i+1)
            create_output_folder(curr_subdirectory) # 카메라별로 subdirectory 생성
            curr_ip = camera_ip[i]['CAM_1']
            token = login(curr_ip, username, password)
            set_white_led(curr_ip, token, 1)
            camera_ip[i]['status'] = 1 # 가동여부 ON
            time.sleep(5)
            snapshot = get_snap(curr_ip, token)
            save_frame(snapshot, curr_subdirectory)
            set_white_led(curr_ip, token, 0)
            camera_ip[i]['status'] = 0 # 가동여부 OFF
            logout(curr_ip, token)
    print(camera_ip)
if __name__ == "__main__":
    main()
