import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from src.model import build_model
from src.music_player import MusicPlayer
from config import *

# Load model
def load_model(weight_path):
    model = build_model(freeze_base=False)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    model.eval()
    return model

# Transform untuk single frame
infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(model, face_img, device):
    tensor = infer_transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred = np.argmax(probs)
    return CLASSES[pred], probs


def run_realtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(weight_path):
        print(" best_model.pth tidak ditemukan! Pastikan training sudah selesai.")
        return

    model = load_model(weight_path).to(device)
    player = MusicPlayer()

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(" Kamera tidak ditemukan!")
        return

    print(" Realtime FER berjalan. Tekan 'Q' untuk keluar.")

    # Smoothing — ambil majority vote dari N frame terakhir
    SMOOTH_N = 5
    recent_preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]

            emotion, probs = predict(model, face_crop, device)

            # Smoothing prediksi
            recent_preds.append(emotion)
            if len(recent_preds) > SMOOTH_N:
                recent_preds.pop(0)
            smoothed_emotion = max(set(recent_preds), key=recent_preds.count)

            # Putar musik sesuai emosi
            player.play(smoothed_emotion)

            # Gambar bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Label emosi
            label = f"{smoothed_emotion.upper()} ({probs.max()*100:.1f}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Bar chart confidence per kelas
            bar_x = 10
            for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
                bar_len = int(prob * 150)
                color = (0, 255, 0) if cls == smoothed_emotion else (100, 100, 100)
                cv2.rectangle(frame, (bar_x, 20 + i*25), (bar_x + bar_len, 40 + i*25), color, -1)
                cv2.putText(frame, f"{cls}: {prob*100:.1f}%", (bar_x + 155, 35 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Kalau ga ada wajah, stop musik
        if len(faces) == 0:
            player.stop()
            recent_preds.clear()
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    player.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(" Selesai.")


if __name__ == "__main__":
    run_realtime()