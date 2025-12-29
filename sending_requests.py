import requests

def send_cough_data():
    url = "http://127.0.0.1:8000/predict"
    file_path = r"D:\PERSONAL PROJECTS\ML Project Ideas\Multimodal Cough-Based Respiratory Health Predictor\Kaggle_Coswara\coswara_wav\20200413\0Rlzhiz6bybk77wdLjxwy7yLDhg1\breathing-deep.wav"  # Update with your actual file path

    try:
        with open(file_path, "rb") as audio_file:
            files = {"file": audio_file}
            response = requests.post(url, files=files)
            response.raise_for_status() 
            response_json = response.json()
            print(response_json)
            
            return response_json
            
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
        
if __name__ == "__main__":
    send_cough_data()
