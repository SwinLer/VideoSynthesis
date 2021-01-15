from tkinter import *
from tkinter import filedialog
from tkinter import BOTH
from tkinter import YES
from PIL import Image, ImageTk
import cv2
import imageio

from SimpleUpdate import update

root = Tk()
root.title('图像合成')
root.geometry('560x500')

front_file = ""
back_file = ""
img1 = Label()
img2 = Label()

def choose_front():
    global front_file, img1
    front_file = filedialog.askopenfilename()
    cap = cv2.VideoCapture(front_file)
    ret, image = cap.read()
    cv2.imwrite("img/front.jpg", image)
    show_image("img/front.jpg", 0.1, 0.35, img1)

def choose_back():
    global back_file, img1
    back_file = filedialog.askopenfilename()
    cap = cv2.VideoCapture(back_file)
    ret, image = cap.read()
    cv2.imwrite("img/back.jpg", image)
    show_image("img/back.jpg", 0.6, 0.35, img1)

def run_convert():
    if len(front_file) < 1 or len(back_file) < 1:
        tip =Label(text="请选择要处理的视频文件！",foreground='red')
        tip.place(relx=0.35, rely=0.05, relwidth=0.3, relheight=0.1)
    else:
        update(front_file, back_file)

def show_image(frame, relx, rely, img):
    imgSize = Image.open(frame)
    tkimage = ImageTk.PhotoImage(imgSize)
    img = Label(image=tkimage)
    img.image = tkimage
    img.place(relx=relx, rely=rely, relwidth=0.3, relheight=0.3)
    #img.bind('<Configure>', handler_adaptor(changeSize, im=imgSize, label=img))
    #img.pack()

# 图片大小自适应
def changeSize(event, im, label):
    image = ImageTk.PhotoImage(im.resize(event.width, event.height), Image.ANTIALIAS)
    label['image'] = image
    label.image = image

def handler_adaptor(fun, **kwds):
    return lambda event, fun=fun, kwds=kwds: fun(event, **kwds)
    
front_btn = Button(root, text="选择前景视频", command=choose_front)
front_btn.place(relx=0.1, rely=0.15, relwidth=0.3, relheight=0.1)

back_btn = Button(root, text="选择背景视频", command=choose_back)
back_btn.place(relx=0.6, rely=0.15, relwidth=0.3, relheight=0.1)

run_btn = Button(root, text="开始视频合成", command=run_convert)
run_btn.place(relx=0.35, rely=0.8, relwidth=0.3, relheight=0.1)

show_image("img/foregroundcover.jpg", 0.1, 0.35, img1)
show_image("img/backgroundcover.jpg", 0.6, 0.35, img2)

root.mainloop()
