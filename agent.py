import subprocess
import requests
import time
import ollama
from PIL import Image
import numpy as np

OLLAMA_URL = "http://localhost:11434"

def is_ollama_running():
    try:
        response = requests.get(OLLAMA_URL)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_daemon():
    # This assumes 'ollama' is in your system PATH
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pull_model(model_name):
    subprocess.run(['ollama', 'pull', model_name])

def wait_for_server(timeout=15):
    for _ in range(timeout):
        if is_ollama_running():
            return True
        time.sleep(1)
    return False
def scale_image_no_interpolation(input_path, output_path, scale_factor):
    """
    Scales an image by an integer factor without interpolation (nearest neighbor).
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the scaled image.
        scale_factor (int): Scaling factor (must be >= 1).
    Returns:
        str: Path to the saved scaled image.
    """
    img = Image.open(input_path)
    img = img.convert("RGB")
    arr = np.array(img)
    scaled_arr = np.repeat(np.repeat(arr, scale_factor, axis=0), scale_factor, axis=1)
    scaled_img = Image.fromarray(scaled_arr)
    scaled_img.save(output_path)
    return output_path
# --- Main automation flow ---
model = 'minicpm-v:8b'
criteria = "You should select the group that eliminates any chance at fully enclosing an area that needs to be covered. This means you should select the group that avoids the possibility that two green areas collide head on. When two green areas collide, they should collide at an angle of at least 90 degrees, not head on. Try to avoid moving down. Also, make sure to leave space directly above any gray area. There should never be a black pixel directly above a gray area. To do this, select the group that is positioned the lowest in the image. Always avoid enclosing or partially enclosing a gray area that needs to be filled. Always leave a gap at the top. The green area that you select will be converted into black, and the adjacent gray pixels will be converted into green. The process will continue until all green areas are covered. If there is no group that satisfies the criteria, select the group that has the least amount of green areas in it. If there is no such group, select the group that is positioned the lowest in the image. Make sure that your response is a valid index of the list provided to you, where the indexes start at 1.  "
system_prompt=f'You are a process planning assistant. You will be shown an image with green areas and your job is to determine which green areas to cover next. {criteria}Your responses should be a single number giving the index of the group. The first index is 1. '


if not is_ollama_running():
    print("Starting Ollama server...")
    start_ollama_daemon()
    if not wait_for_server():
        raise RuntimeError("Ollama server did not start in time.")

print(f"Pulling model '{model}' (if needed)...")
pull_model(model)
messages=[{'role':'system', 'content': system_prompt}]
def get_model_response(user_prompt):
    scale_image_no_interpolation('result.png', 'scaled_result.png', 5)
    messages.append({'role': 'user', 'content': user_prompt, 'images': ['./scaled_result.png']})
    response = ollama.chat(
        model=model,
        messages=messages
    )

    print("Response:", response['message']['content'])
    messages.append({'role': 'assistant', 'content': response['message']['content']})
    if len(messages)>9:
        del messages[1:3]
    return response['message']['content']