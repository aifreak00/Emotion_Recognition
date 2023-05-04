import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import threading
import time
import pygame
from tkinter import *
from PIL import Image, ImageTk

global count
count = 0

def play_sound_thread():
    global count
    pygame.mixer.music.load("sound.mp3")
    pygame.mixer.music.play()
    count = 0

def openCV():
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='C://Users//ASUS//yolov5//last.pt',
                           force_reload=True)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        results = model(frame)
        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

        if (2 in labels):
            global count
            count = count + 1
            if count>1:
                for i in range(1):
                    thread = threading.Thread(target=play_sound_thread)
                    thread.start()

        rendered_frame = np.squeeze(results.render())
        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rendered_frame)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

        root.update()  # Update the root window

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_detection():
    btn_start.config(state=DISABLED)
    btn_stop.config(state=NORMAL)
    thread = threading.Thread(target=openCV)
    thread.start()

def stop_detection():
    btn_stop.config(state=DISABLED)
    btn_start.config(state=NORMAL)
    pygame.mixer.music.stop()

if __name__ == "__main__":
    pygame.mixer.init()
    root = Tk()
    root.title("Drowsiness Detection")
    root.geometry("800x600+200+50")

    # create header label
    header = Label(root, text="Drowsiness Detection", font=("Arial", 24))
    header.pack(pady=20)

    # create image panel
    image = Image.new('RGB', (250,250), (250, 250, 250))
    image = ImageTk.PhotoImage(image)
    panel = Label(root, image=image)
    panel.pack(pady=20)

    # create start button
    btn_start = Button(root, text="Start Detection", font=("Arial", 12), command=start_detection, bg="#000000", fg="white")
    btn_start.pack(pady=10)

    # create stop button
    btn_stop = Button(root, text="Stop Detection", font=("Arial", 12), command=stop_detection, state=DISABLED, bg="#800000", fg="white")
    btn_stop.pack(pady=10)

    def open_settings():
        settings = Toplevel(root)
        settings.geometry("400x300")
        settings.title("Settings")

        threshold_label = Label(settings, text="Alert Threshold:", font=("Arial", 14))
        threshold_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        threshold_entry = Entry(settings, font=("Arial", 14))
        threshold_entry.grid(row=0, column=1, padx=10, pady=10)

        duration_label = Label(settings, text="Alert Duration:", font=("Arial", 14))
        duration_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        duration_entry = Entry(settings, font=("Arial", 14))
        duration_entry.grid(row=1, column=1, padx=10, pady=10)


    root.mainloop()
