import time
import requests

JETSON_IP = '192.168.29.125'  # Replace with Jetson's IP address
URL = f"http://{JETSON_IP}:5000/data"

def send_data():
    while True:
        data = {
            'motion' : 0
        }

        try:
            response = requests.post(URL, json=data)
            if response.status_code == 200:
                print("Data sent successfully:", response.json())
            else:
                print(f"Failed to send data. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Adjust the frequency of sending data

if __name__ == "__main__":
    send_data()
