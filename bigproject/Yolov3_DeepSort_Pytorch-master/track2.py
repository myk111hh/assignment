import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

stop_detection = False

# 初始化摄像头
cap = cv2.VideoCapture("test.mp4")
ret, frame = cap.read()
height, width, _ = frame.shape

# 初始化Tkinter窗口
root = tk.Tk()
root.title("Object Tracking")

# 创建显示视频的画布
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

# 创建用于显示目标跟踪信息的Label
info_label = tk.Label(root, text="", font=("Helvetica", 12))
info_label.pack()

# YOLO配置
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 目标跟踪变量
tracked_object = None
track_color = (255, 255, 0)  # BGR color for tracked object bounding box
boxes = []

def start_tracking(event, x, y, flags, param):
    global tracked_object
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(boxes):
            if x > box[0] and x < box[0] + box[2] and y > box[1] and y < box[1] + box[3]:
                tracked_object = i
                break

cancel_button = tk.Button(root, text="Cancel Tracking", command=lambda: cancel_tracking())
cancel_button.pack()

# 当前被跟踪目标是否已经离开画面
def is_target_out_of_frame(box):
    x, y, w, h = box
    return x < 0 or y < 0 or x + w > width or y + h > height

# 取消跟踪
def cancel_tracking():
    global tracked_object
    tracked_object = None
    info_label.config(text="No Tracked Object")
    boxes.clear()


def update():
    global frame, tracked_object, boxes, stop_detection

    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 仅检测人体类别
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, color, 2)

    cv2.imshow("Camera Feed", frame)
    cv2.setMouseCallback("Camera Feed", start_tracking)

    if tracked_object is not None:
        if tracked_object < len(boxes):
            tracked_box = boxes[tracked_object]
            if is_target_out_of_frame(tracked_box):
                cancel_tracking()

    if tracked_object is not None and tracked_object < len(boxes):
        x, y, w, h = boxes[tracked_object]
        cv2.rectangle(frame, (x, y), (x + w, y + h), track_color, 2)
        info_label.config(text=f"Tracked Object: {classes[class_ids[tracked_object]]}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.img = img

    # 如果按下空格键，停止检测
    key = cv2.waitKey(1)
    if key == ord(' '):
        if stop_detection == True:
            stop_detection = False
        if stop_detection == False:
            stop_detection = True

            # 继续更新循环，除非 stop_detection 为 True
    if not stop_detection:
        root.after(10, update)

# 启动更新循环
update()

root.mainloop()
cap.release()
cv2.destroyAllWindows()

