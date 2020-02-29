import tkinter as tk

from imutils.video import VideoStream
import os
import numpy as np
import imutils
import time
import random
from PIL import ImageTk, Image
import cv2


class ImgData:
    
    data = ['.\me.jpg', '.\shawn.jpg', '.\d1.jpg', '.\d2.jpg', '.\d3.jpg', '.\d4.jpg', '.\d5.jpg']
    pass

def detectVdo():

    print("Loading model...")

    proto = os.path.basename('.\proto.txt')
    model = os.path.basename('.\model.caffemodel')
    confidence = 0.5
    net = cv2.dnn.readNetFromCaffe(proto, model)

    print("Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=720)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confid = detections[0, 0, i, 2]

            if confid < confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confid * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

def detectImg():

    print("Loading model...")

    proto = os.path.basename('.\proto.txt')
    model = os.path.basename('.\model.caffemodel')

    img_file =  random.choice(ImgData.data)
    print(img_file)
    image = os.path.basename(img_file)
    confidence = 0.5
    net = cv2.dnn.readNetFromCaffe(proto, model)

    image = cv2.imread(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


    print("Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confid = detections[0, 0, i, 2]

        if confid > confidence:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confid * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Face Detected Image", image)
    cv2.waitKey(0)

def RGBAImage(path):
	return Image.open(path).convert("RGBA")

def gui():

    root = tk.Tk()
    root.title("Face Detection")
    root.geometry('1920x1080')

    img=RGBAImage(".\\resources\\image.jpg")
    vdo=RGBAImage(".\\resources\\video.jpg")
    bck=RGBAImage(".\\resources\\logo.jpg")
    bck.paste(vdo,(200,170),vdo)
    final_bg=ImageTk.PhotoImage(bck)

    bck.paste(img,(200,570),img)
    final_bg=ImageTk.PhotoImage(bck)

    bg=tk.Label(root,image=final_bg)
    bg.place(x=0,y=0,relwidth=1,relheight=1)

    Button1 = tk.Button(root)
    Button1.place(x=400, y=170, relx=0.067, rely=0.517, height=33, width=78)
    Button1.configure(activebackground="#ececec")
    Button1.configure(activeforeground="#000000")
    Button1.configure(background="#0eb6c9")
    Button1.configure(disabledforeground="#a3a3a3")
    Button1.configure(foreground="#ffffff")
    Button1.configure(highlightbackground="#d9d9d9")
    Button1.configure(highlightcolor="black")
    Button1.configure(pady="0")
    Button1.configure(font="font9")
    Button1.configure(relief="groove")
    Button1.configure(text='''Image''')
    Button1.configure(command=detectImg)
    
    Button2 = tk.Button(root)
    Button2.place(x=400, y=5, relx=0.067, rely=0.25, height=33, width=130)
    Button2.configure(activebackground="#ececec")
    Button2.configure(activeforeground="#000000")
    Button2.configure(background="#0dadbf")
    Button2.configure(disabledforeground="#a3a3a3")
    Button2.configure(foreground="#ffffff")
    Button2.configure(highlightbackground="#d9d9d9")
    Button2.configure(highlightcolor="black")
    Button2.configure(pady="0")
    Button2.configure(font="font9")
    Button2.configure(relief="groove")
    Button2.configure(text='''VideoStream''')
    Button2.configure(command=detectVdo)

    root.mainloop()


if __name__ == '__main__':
    gui()