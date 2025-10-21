# agent.py

import time
import requests
import threading
from pynput import mouse, keyboard

# --- Configuration ---
FLASK_SERVER_URL = "http://127.0.0.1:5000"
USER_ID = "test1" # IMPORTANT: Make sure this matches the user you will register
SEND_INTERVAL = 10

# --- Data Buffers ---
mouse_events = []
keyboard_events = []
lock = threading.Lock()

# --- Event Listeners ---
def on_move(x, y):
    with lock: mouse_events.append({'type': 'move', 'x': x, 'y': y, 'time': time.time()})
def on_click(x, y, button, pressed):
    with lock: mouse_events.append({'type': 'click', 'x': x, 'y': y, 'button': str(button), 'action': 'pressed' if pressed else 'released', 'time': time.time()})
def on_scroll(x, y, dx, dy):
    with lock: mouse_events.append({'type': 'scroll', 'x': x, 'y': y, 'dx': dx, 'dy': dy, 'time': time.time()})
def on_press(key):
    try: key_char = key.char
    except AttributeError: key_char = str(key)
    with lock: keyboard_events.append({'action': 'keydown', 'key': key_char, 'timestamp': time.time() * 1000})
def on_release(key):
    try: key_char = key.char
    except AttributeError: key_char = str(key)
    with lock: keyboard_events.append({'action': 'keyup', 'key': key_char, 'timestamp': time.time() * 1000})

# --- Data Sending Function ---
def send_data_periodically():
    global mouse_events, keyboard_events
    while True:
        time.sleep(SEND_INTERVAL)
        with lock:
            if not mouse_events and not keyboard_events: continue
            payload = {'user_id': USER_ID, 'mouse_events': list(mouse_events), 'keyboard_events': list(keyboard_events)}
            mouse_events.clear(); keyboard_events.clear()
        try:
            print(f"Sending {len(payload['mouse_events'])} mouse and {len(payload['keyboard_events'])} keyboard events...")
            response = requests.post(f"{FLASK_SERVER_URL}/api/authenticate", json=payload)
            if response.status_code == 200: print("Data sent successfully. Server response:", response.json())
            else: print(f"Error sending data: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e: print(f"Could not connect to the server: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting IGRIS Data Collection Agent..."); sender_thread = threading.Thread(target=send_data_periodically, daemon=True); sender_thread.start()
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as m, keyboard.Listener(on_press=on_press, on_release=on_release) as k:
        print("Listeners started. Collecting data..."); m.join(); k.join()