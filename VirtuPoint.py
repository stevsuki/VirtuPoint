import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os, sys
import platform
from tensorflow.keras.models import load_model
import HandTrackModule as htm
import pyautogui, time
from ctypes import cast, POINTER

#function to get resource path for images
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class App(tk.Tk):
    def __init__(self, title, size):
        # Main setup
        super().__init__()
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0], size[1])

        # Back button
        self.back_button = ttk.Button(self, text='Back to Menu', width=12, padding=(0, 5), command=self.back_to_menu)
        self.back_button.place(x=10, y=10)  # Position in the top left corner of the main window
        self.back_button.pack_forget()  # Hide button at the start

        # Widgets
        self.menu = Menu(self)
        self.recognition = Recognition(self)
        self.gesture_information = GestureInformation(self)
        self.about = About(self)

        self.menu.pack()
        
        # Run
        self.mainloop()
        
    def back_to_menu(self):
        self.recognition.stop_video()  # Stop video feed
        # Hide all frames
        self.recognition.pack_forget()
        self.gesture_information.pack_forget()
        self.about.pack_forget()
        
        # Show menu
        self.menu.pack()
        
        # Hide back button
        self.back_button.lower()

class Menu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text='VirtuPoint', anchor='center', font=('Helvetica', 20)).pack(expand=True, fill='both', pady=(100,0))
        self.pack()

        self.create_widgets()

    def create_widgets(self):
        menu_button1 = ttk.Button(self, text="Modul Rekognisi", width=30, padding=(10, 10), command=self.open_recognition, style='TButton')
        menu_button2 = ttk.Button(self, text="Modul Petunjuk", width=30, padding=(10, 10), command=self.open_gesture_information, style='TButton')
        menu_button3 = ttk.Button(self, text="Modul Tentang", width=30, padding=(10, 10), command=self.open_about, style='TButton')

        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12))

        menu_button1.pack(pady=10)
        menu_button2.pack(pady=10)
        menu_button3.pack(pady=10)
    
    def open_recognition(self):
        self.pack_forget()
        self.master.recognition.pack()
        self.master.recognition.tkraise()
        self.master.back_button.tkraise()  # Show back button
        self.master.recognition.start_video()  # Start video feed

    def open_gesture_information(self):
        self.pack_forget()
        self.master.gesture_information.pack()
        self.master.gesture_information.tkraise()
        self.master.back_button.tkraise()  # Show back button
    
    def open_about(self):
        self.pack_forget()
        self.master.about.pack()
        self.master.about.tkraise()
        self.master.back_button.tkraise()  # Show back button

class Recognition(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text='Rekognisi Tangan', anchor='center', font=('Helvetica', 20)).pack(expand=True, fill='both', pady=(15, 15))

        self.widget_frame = ttk.Frame(self, relief='sunken', borderwidth=5)
        self.widget_frame.pack(expand=True, fill='both')

        # Video feed label
        self.video_label = ttk.Label(self.widget_frame)
        self.video_label.pack()

        # Prediction text label
        self.prediction_label = ttk.Label(self, text='', font=('Helvetica', 14))
        self.prediction_label.pack(pady=(10, 0))  # Position it below the video feed

        # Initialize parameters
        self.img_size = 100
        self.labels = ['Control Volume', 'Drag Mouse', 'Mouse Click', 'Moving Cursor', 'Right Click', 'not detected']
        self.predict_interval = 5
        self.frame_count = 0
        self.last_predicted_label = self.labels[5]
        self.accuracy = 0
        self.no_detection_count = 0
        self.pTime = 0
        self.tipId = [8, 12, 16, 20]
        self.frameR = 80
        self.smoothening = 7
        self.wScreen, self.hScreen = pyautogui.size()
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.wCam  = 500 
        self.hCam = 400
        self.cap = None
        self.model = load_model(resource_path("best_custom_cnn_model_final.keras"))
        self.detector = htm.handDetector(maxHands=1)
        self.detect_enabled = False  # Track whether detection is enabled
        self.detect_enabled_volume = False  # Track whether volume control detection is enabled
        self.detect_enabled_recognition = False  # Track whether gesture recognition detection is enabled
        self.os_type = platform.system() 
        if self.os_type == 'Windows':
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
            #get only 2 value return max and min from getVloumeRange
            self.vol_min, self.vol_max = self.volume_control.GetVolumeRange()[:2]
        self.min_distance = 50
        self.max_distance = 150

        # Bind the key event
        self.bind_all('<KeyPress-e>', self.enable_detection)

    def enable_detection(self, event):
        self.detect_enabled = not self.detect_enabled

    def start_video(self):
        self.cap = cv2.VideoCapture(0)  # Use the default camera (0)
        self.update_video()  # Start video feed

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()  # Release the video capture
            self.cap = None  # Set to None to indicate it's stopped
        self.video_label.imgtk = None  # Clear the video label
        self.detect_enabled = False  # Reset detection flag

    def update_video(self):
        if self.cap is None:
            return  # Exit if video capture is stopped
        success, img = self.cap.read()
        if success:
            img = cv2.resize(img, (self.wCam, self.hCam))
            imgHand, bbox, typeHand = self.detector.findHands(img, padding=15)
            lmlist = self.detector.findPositions(imgHand, draw=False)

            if self.detect_enabled:
                if not bbox:
                    # No hand detected
                    self.prediction_label.config(text='Please show your hand for detection')
                else:
                    # Hand detected, proceed with prediction
                    x, y, w, h = bbox[0]
                    imgCrop = img[y:y + h, x:x + w]
                    imgWhite = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = self.img_size / h
                        wCal = int(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, self.img_size))
                        wGap = (self.img_size - wCal) // 2
                        imgWhite[:, wGap:wGap + wCal] = imgResize
                    else:
                        k = self.img_size / w
                        hCal = int(k * h)
                        imgResize = cv2.resize(imgCrop, (self.img_size, hCal))
                        hGap = (self.img_size - hCal) // 2
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    imgWhiteGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.medianBlur(imgWhiteGray, 5)
                    _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
                        
                    imgWhiteGray = np.expand_dims(thresholded_image, axis=-1) / 255.0
                    imgWhiteGray = np.expand_dims(imgWhiteGray, axis=0)

                    fingers = self.detector.fingersUp(lmlist, typeHand)
                    print(fingers)

                    isAllFingerDown = all(finger == 0 for finger in fingers[1:6])
                    isAllFingerUp = all(finger == 1 for finger in fingers[1:6])

                    if (self.last_predicted_label == "Control Volume" and isAllFingerDown) or (not self.last_predicted_label == "Control Volume" and (isAllFingerDown or isAllFingerUp)):
                        self.freeze_prediction = True
                    else:
                        self.freeze_prediction = False

                    predicted_class = ""

                    if not self.freeze_prediction:
                        if self.frame_count % self.predict_interval == 0:
                            predictions = self.model.predict(imgWhiteGray)
                            predicted_class = np.argmax(predictions[0])
                            self.last_predicted_label = self.labels[predicted_class]
                            self.accuracy = predictions[0][predicted_class] * 100

                    if not self.freeze_prediction:
                        predicted_label = f"{self.last_predicted_label} ({self.accuracy:.2f}%)"
                    else:
                        self.last_predicted_label = "Gesture not recognized. Try another gesture"
                        predicted_label = self.last_predicted_label

                    self.prediction_label.config(text=predicted_label)

                    x1, y1 = lmlist[8][1:]
                    x2, y2 = lmlist[4][1:]
                    
                    x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScreen))
                    y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScreen))

                    # smoothen values
                    self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                    self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening

                    if self.last_predicted_label == "Moving Cursor":
                        cv2.rectangle(img, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR), (255, 255, 0), 2)
                        # move mouse
                        pyautogui.moveTo(self.wScreen - self.clocX, self.clocY, _pause=False)
                        self.plocX, self.plocY = self.clocX, self.clocY

                    if self.last_predicted_label == "Mouse Click":
                        pyautogui.click()

                    if self.last_predicted_label == "Right Click":
                        pyautogui.click(button="right")
                        
                    if self.last_predicted_label == "Drag Mouse":
                        pyautogui.mouseDown()

                    if self.last_predicted_label == "Control Volume":
                        # Adjust volume
                        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        distance_clamped = np.clip(distance, self.min_distance, self.max_distance)

                        if self.os_type == 'Windows':
                            # Interpolasi volume Windows
                            volume = np.interp(distance_clamped, [self.min_distance, self.max_distance], [self.vol_min, self.vol_max])
                            self.volume_control.SetMasterVolumeLevel(volume, None)
                            self.volume_percentage = int(np.interp(volume, [self.vol_min, self.vol_max], [0, 100]))
                        elif self.os_type == 'Linux':
                            # Interpolasi volume Linux (0-100%)
                            self.volume_percentage = int(np.interp(distance_clamped, [self.min_distance, self.max_distance], [0, 100]))
                            os.system(f"amixer -D pulse sset Master {self.volume_percentage}%")
                        elif self.os_type == 'Darwin':  # Darwin adalah nama platform untuk macOS
                            # Interpolasi volume macOS (0-100%)
                            self.volume_percentage = int(np.interp(distance_clamped, [self.min_distance, self.max_distance], [0, 100]))
                            os.system(f"osascript -e 'set volume output volume {self.volume_percentage}'")

                        cv2.line(img, (x2, y2), (x1, y1), (0, 255, 0), 3)
                        cv2.putText(img, f'Volume: {self.volume_percentage}%', 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                # Detection is disabled
                self.prediction_label.config(text='')

            # Display the video feed
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.frame_count += 1
        self.video_label.after(10, self.update_video)

    def __del__(self):
        self.stop_video()  # Ensure video is released
        cv2.destroyAllWindows()

class GestureInformation(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text='Petunjuk Gesture', anchor='center', font=('Helvetica', 20)).pack(expand=True, fill='both', pady=(15, 15))

        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill='both')

        self.widget_frame = ttk.Frame(main_frame, relief='sunken', borderwidth=5)
        self.widget_frame.pack(side='left', expand=True, fill='both')

        # Menambahkan gambar dan deskripsi menggunakan create_image_info
        self.create_image_info(resource_path('images/moving_cursor.jpg'), 'Moving Cursor Hand Gesture')
        self.create_image_info(resource_path('images/mouse_click.jpg'), 'Mouse Click Hand Gesture')
        self.create_image_info(resource_path('images/drag_mouse.jpg'), 'Mouse Drag Hand Gesture')
        self.create_image_info(resource_path('images/control_volume.jpg'), 'Control Volume Gesture')

        self.widget_frame2 = ttk.Frame(main_frame, relief='sunken', borderwidth=5)
        self.widget_frame2.pack(side='right', expand=True, fill='both')

        self.create_image_info(resource_path('images/right_click.jpg'), 'Right Click Gesture', self.widget_frame2)

        desc = """
                Virtual Mouse Controls Keyboard Key:


                  Press <E> to Start Detecting Hand
            """
        description_label = ttk.Label(self.widget_frame2, text=desc, anchor='w', justify='left', font=('Helvetica', 11))
        description_label.pack(side='left')

        frame = ttk.Frame(self.widget_frame2)
        frame.pack(side='top', fill='x', pady=0, padx=30) 


    def create_image_info(self, imagePath, desc, parent_frame=None):
        if parent_frame is None:
            parent_frame = self.widget_frame
            
        frame = ttk.Frame(parent_frame)
        frame.pack(side='top', fill='x', pady=10, padx=10)  # Mengatur agar tersusun ke bawah

        imageInfo = Image.open(imagePath)
        imageInfo = imageInfo.resize((100, 100))
        photo = ImageTk.PhotoImage(imageInfo)

        box_frame = ttk.Label(frame, image=photo)
        box_frame.image = photo  # Menyimpan referensi gambar
        box_frame.pack(side='left')  # Menyusun gambar ke kiri di dalam frame

        # Menambahkan deskripsi di sebelah kanan gambar
        description_label = ttk.Label(frame, text=desc, anchor='w', justify='left', font=('Helvetica', 12))
        description_label.pack(side='left', padx=10)  # Menyusun deskripsi ke kanan gambar

class About(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text='Tentang Aplikasi', anchor='center', font=('Helvetica', 20)).pack(expand=True, fill='both', pady=(15, 15))

        self.widget_frame = ttk.Frame(self, relief='sunken', borderwidth=3)
        self.widget_frame.pack(expand=True, fill='both')

        desc = """
                                                                  Virtual Mouse Application

Aplikasi ini dirancang untuk memberikan pengalaman kontrol komputer yang intuitif menggunakan gerakan tangan. Dengan memanfaatkan teknologi pengenalan gambar dan deteksi tangan, aplikasi ini memungkinkan pengguna untuk mengendalikan kursor, mengklik, drag dan mengatur volume hanya dengan gerakan tangan.

Fitur Utama:
    - Kontrol Kursor: Menggerakkan kursor dengan gerakan tangan.
    - Klik Mouse: Menggunakan gerakan tangan untuk melakukan klik.
    - Drag Mouse: Menggunakan gerakan tangan untuk melakukan drag mouse.
    - Pengaturan Volume: Mengatur volume suara dengan jarak antara dua jari.
    - Deteksi Bentuk Gestur Tangan: Menggunakan model CNN untuk mengenali berbagai gerakan tangan.

    
                                                                   Dibuat oleh: Steven Suki
                                                      Pembimbing: Dra. Chairisni Lubis, M.Kom.
                                                            Program Studi Teknik Informatika
                                                                  Universitas Tarumanagara 
                                                                                ©2024
                                                                

"""

        self.box_frame = ttk.Label(self.widget_frame, text=desc, wraplength=700, justify='left')
        self.box_frame.pack()

if __name__ == "__main__":
    app = App("Virtual Mouse", (800, 600))
