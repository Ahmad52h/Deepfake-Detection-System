import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os

# --- 1. تحميل النظام الأساسي (العقل + العيون) ---
print("⏳ Loading AI brain and detection tools...")
try:
    # تحميل العقل الذي دربناه
    model = tf.keras.models.load_model("cloud_deepfake_model_autosave.h5")
    # 💡 This is the line that was missing! (Defining the face detection tool)
    detector = MTCNN() 
    print("✅ The system has been loaded successfully! Running the interface...")
except Exception as e:
    print(f"❌ Error loading the system: {e}")
    # If the model is not found, we will run the detector at least so that the program does not crash
    detector = MTCNN()

# --- 2. The Smart and Secure Analysis Function ---
def analyze_video(video_path, progress=gr.Progress()):
    if not video_path:
        return " ❌ Please upload a video first."
    
    print(f"\n📂 Reading video from path: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return " ❌ Technical Problem: The system could not read the video file. Try an MP4 video."

    frames = []
    frame_count = 0
    
    progress(0.3, desc="Extracting faces and analyzing...")
    while cap.isOpened() and len(frames) < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # استخدام الكاشف الذي عرفناه في الأعلى
                results = detector.detect_faces(rgb_frame)
                if results:
                    x, y, w, h = results[0]['box']
                    x, y = max(0, x), max(0, y)
                    face = frame[y:y+h, x:x+w]
                    
                    if face.size > 0: # التأكد أن الوجه سليم
                        face_resized = cv2.resize(face, (224, 224))
                        # 💡 أزلنا القسمة على 255، وحولناها لنفس النوع الذي تدرب عليه!
                        face_normalized = face_resized.astype(np.float16) 
                        frames.append(face_normalized)
            except Exception as e:
                pass # تجاهل الإطارات المعطوبة بصمت
                
        frame_count += 1
    cap.release()

    if len(frames) < 10:
        return f"⚠️ Video does not contain clear enough faces. (System found {len(frames)} frames and needs 10)."

    progress(0.8, desc="The AI is making the final decision")
    
    try:
        sequence = np.expand_dims(np.array(frames), axis=0)
        prediction = model.predict(sequence)[0][0]
        
        if prediction >= 0.5:
            confidence = prediction * 100
            return f"🚨 Warning: This video is a deepfake! AI verification percentage: {confidence:.2f}%"
        else:
            confidence = (1 - prediction) * 100
            return f"✅Safety: This video is real and authentic. AI verification level: {confidence:.2f}%"
    except Exception as e:
        return f"❌An internal error occurred in the decision-making process: {e}"

# --- 3. تصميم الواجهة ---
interface = gr.Interface(
    fn=analyze_video, 
    inputs=gr.Video(label="Upload or record the video here"), 
    outputs=gr.Textbox(label="The final result (The decision)", lines=3),
    title="🕵️‍♂️ Deepfake Detector System",
    description="Upload any video containing a human face. The system will extract faces, analyze motion consistency, and detect any manipulation using (CNN + RNN) techniques."
)

if __name__ == "__main__":
    interface.launch(theme="soft")