import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2

model_path = 'saved_model'
model = tf.saved_model.load(model_path)

labels_path = 'label_map.pbtxt'
labels = {}
with open(labels_path, 'r') as f:
    current_id = None
    for line in f:
        if "id" in line:
            current_id = int(line.split(':')[-1].strip())
        if "name" in line:
            labels[current_id] = line.split(':')[-1].strip().replace("'", "")

def draw_boxes(image, boxes, classes, scores, labels, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            class_id = classes[i]
            label = labels.get(class_id, 'Unknown')

            h, w, _ = image.shape
            y_min, x_min, y_max, x_max = box
            y_min, x_min, y_max, x_max = int(y_min * h), int(x_min * w), int(y_max * h), int(x_max * w)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def perform_detection(image_path):
    image = cv2.imread(image_path)
    input_tensor = cv2.resize(image, (320, 320))
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    draw_boxes(image, detection_boxes, detection_classes, detection_scores, labels)

    return image

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            analyzed_image = perform_detection(file_path)
            analyzed_image_rgb = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(analyzed_image_rgb)
            img.thumbnail((600, 600))
            img = ImageTk.PhotoImage(img)
            panel.config(image=img)
            panel.image = img
            status_label.config(text="Objects detected!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            status_label.config(text="Error analyzing image")

root = tk.Tk()
root.title("Object Detection")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn_browse = tk.Button(frame, text="Choose Image", command=browse_image, padx=10, pady=5, bg="#4CAF50", fg="white")
btn_browse.pack(side=tk.LEFT)

panel = tk.Label(root)
panel.pack(padx=20, pady=20)

status_label = tk.Label(root, text="Select an image to analyze", pady=10)
status_label.pack()

root.mainloop()
