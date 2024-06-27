
import streamlit as st
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import os
import gdown       
import vonage
import time
import threading
import queue

# Initialize Vonage client, YOLO model, and other setups as before
client = vonage.Client(key="12fc1bd8", secret="sKVeBad8vSgVGdrp")
sms = vonage.Sms(client)

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

folder_id = "1w2-zNG0mUt6SufkO1YvoC-KsqQX7DNaR"
local_directory = r"C:\Users\asuss\OneDrive\Pictures"
os.makedirs(local_directory, exist_ok=True)

gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=local_directory)

# Use queues for thread-safe communication
scheduled_runs_queue = queue.Queue()
completed_runs_queue = queue.Queue()

def run_script(activity_name):
    face_detected_images = []
    face_detected = False

    for filename in os.listdir(local_directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(local_directory, filename)
            image = Image.open(image_path)
            output = model(image)
            results = Detections.from_ultralytics(output[0])

            if len(results) > 0:
                face_detected = True
                face_detected_images.append(image_path)

    if face_detected:
        responseData = sms.send_message(
            {
                "from": "Vonage APIs",
                "to": "919698095155",
                "text": f"A face was detected in one or more images for activity: {activity_name}",
            }
        )
        print(f"Activity '{activity_name}' completed. Face detected in {len(face_detected_images)} images.")
    else:
        print(f"Activity '{activity_name}' completed. No faces detected.")
    
    return face_detected_images

def schedule_checker():
    while True:
        current_time = datetime.now()
        scheduled_runs = list(scheduled_runs_queue.queue)
        for scheduled_time, activity_name in scheduled_runs:
            if current_time >= scheduled_time:
                detected_images = run_script(activity_name)
                scheduled_runs_queue.get()  # Remove the completed run
                completed_runs_queue.put((activity_name, detected_images))
        time.sleep(60)  # Check every minute

def main():
    st.title("Web Monitor")

    # Check Activity Now button
    if st.button("Check Activity Now"):
        detected_images = run_script("Manual Check")
        if detected_images:
            st.write(f"Faces detected in Manual Check:")
            for img_path in detected_images:
                st.image(img_path, caption=f"Face detected in {os.path.basename(img_path)}")
        else:
            st.write("No faces detected in Manual Check.")

    # Display results of completed scheduled runs
    while not completed_runs_queue.empty():
        activity_name, detected_images = completed_runs_queue.get()
        if detected_images:
            st.write(f"Faces detected in scheduled activity: {activity_name}")
            for img_path in detected_images:
                st.image(img_path, caption=f"Face detected in {os.path.basename(img_path)}")
        else:
            st.write(f"No faces detected in scheduled activity: {activity_name}")

    # Scheduling options
    st.write("Schedule New Activity")
    schedule_option = st.radio("Choose scheduling option:", ("Specific Date and Time", "Daily at Specific Time"))
    
    if schedule_option == "Specific Date and Time":
        run_date = st.date_input("Select date")
        run_time = st.time_input("Select time")
        schedule_time = datetime.combine(run_date, run_time)
    else:
        run_time = st.time_input("Select daily time")
        schedule_time = datetime.combine(datetime.now().date(), run_time)
        if schedule_time <= datetime.now():
            schedule_time += timedelta(days=1)

    activity_name = st.text_input("Enter activity name")

    if st.button("Schedule Activity"):
        if schedule_time > datetime.now():
            scheduled_runs_queue.put((schedule_time, activity_name))
            st.success(f"Activity '{activity_name}' scheduled for {schedule_time}")
        else:
            st.error("Please select a future time.")

    # Display upcoming scheduled runs
    st.write("Upcoming Scheduled Runs:")
    for scheduled_time, activity in list(scheduled_runs_queue.queue):
        st.write(f"- {scheduled_time.strftime('%Y-%m-%d %H:%M')} - {activity}")

if __name__ == "__main__":
    # Start the schedule checker in a separate thread
    threading.Thread(target=schedule_checker, daemon=True).start()
    main()
