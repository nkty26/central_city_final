import requests, json, time, requests
from datetime import datetime
import cv2

class Acquisition:
    def __init__(self, config_path='/home/asiadmin/Workspace/CENTRAL_FINAL/config_ipcam.json', save_dir="/home/asiadmin/Workspace/CENTRAL_FINAL/DATASETS/IMAGES/"):
        self.headers = {"Content-Type": "application/json"}
        self.config_path = config_path
        self.save_dir = save_dir
        self.camera_data = self.load_config()
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            return config['camera_data']

    def login(self, camera_data, username, password):
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
        
        response = requests.post(url=url, headers=self.headers, data=json.dumps(data))
        result = response.json()
        TOKEN = result[0]["value"]["Token"]["name"]
        if response.status_code == 200:
            print(f" --> IP CAM Login Successful")
        return TOKEN

    def set_white_led(self, camera_data, token, state):
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
        response = requests.post(url=url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200 and state == 0:
            print(f" --> Turned off the IR LED")
        elif response.status_code == 200 and state == 1:
            print(f" --> Turned on the IR LED")
        else:
            print(f" --> IR LED Error")

    def get_snap(self, camera_data, token):
        url = f'http://{camera_data}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=flsYJfZgM6RTB_os&token={token}'
        response = requests.post(url=url, headers=self.headers)
        if response.status_code == 200:
            print(f" --> Snapshot Successful")
        return response.content

    def create_output_folder(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f" --> Created folder: {output_folder}")

    def save_frame(self, frame, save_dir):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f'{timestamp_str}.jpg'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(frame)
        return filepath

    def logout(self, camera_data, token):
        url = f'http://{camera_data}/api.cgi?cmd=Logout&token={token}'
        data = [{
            "cmd": "Logout",
            "param": {}
        }]
        response = requests.post(url=url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            print(f" --> Logout Successful")

    def render(self, image, string):
        cv2.imshow(string, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def acquire_data(self):
        results = [{} for _ in range(len(self.camera_data))]
        self.create_output_folder(self.save_dir)
        for i in range(len(self.camera_data)):
            curr_cam = self.camera_data[i]['name']
            if curr_cam == "CAM_1":
                password = self.camera_data[i]['password']
            username = self.camera_data[i]['username']
            if self.camera_data[i]['status'] == 0:
                curr_subdirectory = os.path.join(self.save_dir, curr_cam)
                self.create_output_folder(curr_subdirectory)
                curr_ip = self.camera_data[i]['ip']
                token = self.login(curr_ip, username, password)
                self.set_white_led(curr_ip, token, 1)
                self.camera_data[i]['status'] = 1
                time.sleep(2)
                snapshot = self.get_snap(curr_ip, token)
                saved_image = self.save_frame(snapshot, curr_subdirectory)
                self.set_white_led(curr_ip, token, 0)
                self.camera_data[i]['status'] = 0
                self.logout(curr_ip, token)
                results[i].update({"curr_cam": curr_cam, "acquisition_path": saved_image})
            else:
                print(f"inactive camera. set camera {curr_cam} status to 0")
                continue
        return results

def main():
    acquisition = Acquisition()
    acquisition.acquire_data()

if __name__ == '__main__':
    main()
