import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import playsound
import tempfile
import threading
import queue
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import time
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    exit()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    max_num_hands=1
)

# Speech queue and lock
speech_queue = queue.Queue()
speech_lock = threading.Lock()
is_speaking = False

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: ' ',
    37: '.'
}
expected_features = 42

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""

def remove_last_letter():
    """Remove the last letter from the current word"""
    global word_buffer
    if word_buffer:
        word_buffer = word_buffer[:-1]
        current_word.set(word_buffer if word_buffer else "N/A")
        # Update sentence display without the last letter
        if sentence:
            current_sentence.set((sentence + " " + word_buffer) if word_buffer else sentence.strip())
        else:
            current_sentence.set(word_buffer if word_buffer else "")

def text_to_speech(text):
    """Convert text to speech using gTTS and playsound"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            playsound.playsound(fp.name)
            os.unlink(fp.name)  # Delete the temporary file
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def speech_worker():
    """Worker thread to process speech queue"""
    global is_speaking
    while True:
        text = speech_queue.get()
        if text is None:  # Exit signal
            break
        try:
            with speech_lock:
                is_speaking = True
                text_to_speech(text)
                is_speaking = False
        except Exception as e:
            print(f"Error in speech worker: {e}")
            is_speaking = False
        finally:
            speech_queue.task_done()

def speak_text(text):
    """Add text to speech queue if not empty and not already speaking"""
    if not text.strip():
        return
    with speech_lock:
        if not is_speaking:
            speech_queue.put(text)

# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# GUI Setup
root = tk.Tk()
root.title("Sign Language to Speech Conversion")

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window to fullscreen
root.attributes('-fullscreen', True)

# Alternatively, if you want to maximize but not fullscreen:
# root.state('zoomed')

root.configure(bg="#2c2f33")

# Variables for GUI
current_alphabet = StringVar(value="Initializing...")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="")
is_paused = StringVar(value="False")

# Title
title_label = Label(root, text="Sign Language to Speech Conversion", 
                   font=("Arial", 28, "bold"), fg="#ffffff", bg="#2c2f33")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=640, height=480)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)

content_frame = Frame(root, bg="#2c2f33", width=580, height=400)
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))
content_frame.grid_propagate(True)

button_frame = Frame(root, bg="#2c2f33", height=100)
button_frame.grid(row=3, column=1, pady=(10, 20), padx=(10, 20), sticky="n")
button_frame.grid_propagate(True)

# Video feed
video_label = tk.Label(video_frame, text="Loading camera...", 
                      font=("Arial", 16), fg="white", bg="black")
video_label.pack(expand=True)

# Labels
Label(content_frame, text="Current Alphabet:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(0, 10))
Label(content_frame, textvariable=current_alphabet, font=("Arial", 24, "bold"), 
      fg="#1abc9c", bg="#2c2f33").pack(anchor="center")

Label(content_frame, text="Current Word:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), 
      fg="#f39c12", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Current Sentence:", font=("Arial", 20), 
      fg="#ffffff", bg="#2c2f33").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), 
      fg="#9b59b6", bg="#2c2f33", wraplength=500, justify="left").pack(anchor="center")

def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("N/A")
    current_sentence.set("")
    current_alphabet.set("N/A")

def toggle_pause():
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Resume")
    else:
        is_paused.set("False")
        pause_button.config(text="Pause")

def toggle_fullscreen(event=None):
    root.attributes('-fullscreen', not root.attributes('-fullscreen'))

def exit_fullscreen(event=None):
    root.attributes('-fullscreen', False)

# Buttons
# Button(button_frame, text="Reset Sentence", font=("Arial", 16), 
#        command=reset_sentence, bg="#e74c3c", fg="red", 
#        relief="flat", height=2, width=14).grid(row=0, column=0, padx=10)

# pause_button = Button(button_frame, text="Pause", font=("Arial", 16), 
#                      command=toggle_pause, bg="#3498db", fg="black", 
#                      relief="flat", height=2, width=12)
# pause_button.grid(row=0, column=1, padx=10)

# speak_button = Button(button_frame, text="Speak Sentence", font=("Arial", 16), 
#                       command=lambda: speak_text(current_sentence.get()), 
#                       bg="#27ae60", fg="black", relief="flat", height=2, width=14)
# speak_button.grid(row=0, column=2, padx=10)

Button(button_frame, text="Remove Last Letter", font=("Arial", 16), 
       command=remove_last_letter, bg="#e67e22", fg="black", 
       relief="flat", height=2, width=14).grid(row=0, column=0, padx=10)

Button(button_frame, text="Reset Sentence", font=("Arial", 16), 
       command=reset_sentence, bg="#e74c3c", fg="black", 
       relief="flat", height=2, width=14).grid(row=0, column=1, padx=10)

pause_button = Button(button_frame, text="Pause", font=("Arial", 16), 
                     command=toggle_pause, bg="#3498db", fg="black", 
                     relief="flat", height=2, width=12)
pause_button.grid(row=0, column=2, padx=10)

speak_button = Button(button_frame, text="Speak Sentence", font=("Arial", 16), 
                      command=lambda: speak_text(current_sentence.get()), 
                      bg="#27ae60", fg="black", relief="flat", height=2, width=14)
speak_button.grid(row=0, column=3, padx=10)

# Add fullscreen toggle with F11 key
root.bind('<F11>', toggle_fullscreen)
root.bind('<Escape>', exit_fullscreen)

# Video Capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    current_alphabet.set("Ready")
except Exception as e:
    messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
    video_label.config(text="Camera Error")
    cap = None

def process_frame():
    start_time = time.time()

    if not cap or not cap.isOpened():
        return

    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        video_label.config(text="Camera Feed Error")
        return

    if is_paused.get() == "True":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure valid data
            data_aux = data_aux[:expected_features] + [0] * (expected_features - len(data_aux))

            # Predict gesture
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            # Stabilization logic with smaller buffer
            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 15:
                stabilization_buffer.pop(0)

            if stabilization_buffer.count(predicted_character) > 12:
                current_time = time.time()
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time
                    current_alphabet.set(stable_char)

                    # Handle word and sentence formation
                    if stable_char == ' ':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + " "
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    elif stable_char == '.':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + "."
                            current_sentence.set(sentence.strip())
                        word_buffer = ""
                        current_word.set("N/A")
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)
                        current_sentence.set((sentence + " ") + word_buffer if sentence else word_buffer)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        latency = (time.time() - start_time) * 1000  # Calculate in milliseconds
        print(f"Frame processed in {latency:.2f}ms")  # Log to console

    # Draw UI elements
    cv2.putText(frame, f"Alphabet: {current_alphabet.get()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'Pause' to freeze frame", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Update video feed
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

def on_closing():
    """Cleanup on window close"""
    if cap and cap.isOpened():
        cap.release()
    speech_queue.put(None)
    speech_thread.join()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Variables for stabilization timing
last_registered_time = time.time()
registration_delay = 1.0

if cap and cap.isOpened():
    process_frame()
else:
    speak_button.config(state="disabled")

root.mainloop()
