import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy import signal


class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        
        self.freq_attn = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.3)
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        attn = self.freq_attn(x)
        x = (x * attn).sum(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class BirdClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bird Sound Classifier")
        self.root.geometry("500x750")
        self.root.resizable(False, False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        
        if self.model is None:
            messagebox.showerror("Không thể tải model")
            self.root.quit()
            return
            
        self.class_names = self.get_class_names()
        self.current_bird_image = None
        self.setup_background()

        self.result_frame = tk.Frame(self.root, bg="white", relief=tk.RAISED, bd=2)
        self.result_frame.place(x=50, y=50, width=400, height=400)
        
        self.image_label = tk.Label(
            self.result_frame,
            bg="white",
            text=""
        )
        self.image_label.pack(pady=(20, 10))
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Kết quả sẽ hiển thị ở đây",
            font=("Arial", 14),
            bg="white",
            fg="#7f8c8d",
            wraplength=360,
            justify=tk.CENTER
        )
        self.result_label.pack(expand=True, padx=20, pady=(10, 20))
        
        self.mic_button = self.create_round_mic_button(self.root)
        self.mic_button.place(x=190, y=500)

        self.status_label = tk.Label(
            root,
            text="Sẵn sàng",
            font=("Arial", 9),
            bg="#ecf0f1",
            fg="#7f8c8d",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        if not os.path.exists(model_path):
            messagebox.showerror("Lỗi", "Không tìm thấy file model:\nbest_model.pth")
            return None
        
        model = ImprovedCRNN(num_classes=22).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def get_class_names(self):
        return [
            "Fringillidae_Serinus_canicollis",
            "Fringillidae_Serinus_serinus",
            "Paridae_Hypocnemis_cantator",
            "Paridae_Hypocnemis_hypoxantha",
            "Paridae_Hypocnemis_peruviana",
            "Paridae_Hypocnemis_striata",
            "Paridae_Saxicola_gutturalis",
            "Paridae_Saxicola_rubetra",
            "Paridae_Saxicola_rubicola",
            "Paridae_Saxicola_tectes",
            "Paridae_Saxicola_torquatus",
            "Troglodytidae_Troglodytes_aedon",
            "Troglodytidae_Troglodytes_hiemalis",
            "Troglodytidae_Troglodytes_pacificus",
            "Troglodytidae_Troglodytes_troglodytes",
            "Turdidae_Catharus_aurantiirostris",
            "Turdidae_Catharus_bicknelli",
            "Turdidae_Catharus_fuscater",
            "Turdidae_Catharus_fuscescens",
            "Turdidae_Catharus_guttatus",
            "Turdidae_Catharus_minimus",
            "Turdidae_Catharus_ustulatus"
        ]
    
    def detect_bird_segment(self, y, sr, target_duration=3.0, hop_length=512):
        # Lọc tần số chim (1000-8000 Hz) bằng bandpass filter
        nyquist = sr / 2
        low = 1000 / nyquist
        high = min(8000 / nyquist, 0.99)
        
        # Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        y_filtered = signal.filtfilt(b, a, y)
        
        # Tính năng lượng của từng frame
        frame_length = int(sr * 0.1)  # 100ms frames
        hop = frame_length // 2
        
        energy = np.array([
            np.sum(y_filtered[i:i+frame_length]**2)
            for i in range(0, len(y_filtered) - frame_length, hop)
        ])
        
        # Smooth năng lượng bằng moving average
        window = 5
        energy_smooth = np.convolve(energy, np.ones(window)/window, mode='same')
        
        # Tìm đoạn có năng lượng cao nhất
        target_samples = int(target_duration * sr)
        target_frames = target_samples // hop
        
        if len(energy_smooth) < target_frames:
            # Audio quá ngắn, lấy toàn bộ
            return y
        
        # Tính tổng năng lượng của mỗi cửa sổ target_duration
        window_energy = np.array([
            np.sum(energy_smooth[i:i+target_frames])
            for i in range(len(energy_smooth) - target_frames + 1)
        ])
        
        # Vị trí có năng lượng cao nhất
        best_idx = np.argmax(window_energy)
        start_sample = best_idx * hop
        end_sample = start_sample + target_samples
        
        # Đảm bảo không vượt quá độ dài
        end_sample = min(end_sample, len(y))
        
        return y[start_sample:end_sample]
    
    def extract_logmel(self, y, sr=48000, n_fft=2048, hop_length=512, n_mels=128, fmin=1000, fmax=8000):
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    
    def fix_time_dim(self, spec, target_frames):
        T = spec.shape[1]

        if T == target_frames:
            return spec

        if T < target_frames:
            pad = target_frames - T
            return np.pad(spec, ((0, 0), (0, pad)), mode="constant")

        start = (T - target_frames) // 2
        return spec[:, start:start + target_frames]
    
    def preprocess_audio(self, audio_path):
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=48000, mono=True)
            
            if len(y) == 0:
                print("File âm thanh rỗng")
                return None
            
            # Cắt đoạn có tiếng chim 3 giây
            print(f"Audio length: {len(y)/sr:.2f}s, detecting bird segment...")
            y_segment = self.detect_bird_segment(y, sr, target_duration=3.0)
            print(f"Selected segment: {len(y_segment)/sr:.2f}s")
            
            # Extract log-mel spectrogram
            spec = self.extract_logmel(y_segment, sr=sr)
            
            # Fix time dimension
            target_frames = 282 # tương ứng 3 giây
            spec = self.fix_time_dim(spec, target_frames)
            
            # Normalize
            mean = spec.mean()
            std = spec.std()
            if std == 0:
                std = 1.0
            spec = (spec - mean) / (std + 1e-6)

            spec = np.clip(spec, -6.0, 6.0)

            # Convert to tensor
            spec_tensor = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            return spec_tensor.to(self.device)
            
        except Exception as e:
            print(f"Lỗi xử lý file: {e}")
            return None
    
    def classify_audio(self, audio_path):
        if self.model is None:
            return "Model chưa tải", 0.0
        
        input_tensor = self.preprocess_audio(audio_path)
        if input_tensor is None:
            return "Lỗi xử lý âm thanh", 0.0
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        class_name = self.class_names[predicted_idx.item()]
        confidence = confidence.item() * 100
        
        return class_name, confidence

    def setup_background(self):
        bg_path = os.path.join(os.path.dirname(__file__), "images", "background.jpg")
        if os.path.exists(bg_path):
            bg_image = Image.open(bg_path).resize((500, 750), Image.Resampling.LANCZOS)
            bg_image = ImageEnhance.Brightness(bg_image).enhance(0.4)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            tk.Label(self.root, image=self.bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

    def create_round_mic_button(self, parent):
        button_size = 120
        canvas = tk.Canvas(parent, width=button_size, height=button_size, highlightthickness=0)
        self.circle = canvas.create_oval(5, 5, button_size-5, button_size-5,
                                         fill="#e74c3c", outline="#c0392b", width=3)
        
        icon_path = os.path.join(os.path.dirname(__file__), "images", "micbird.png")
        if os.path.exists(icon_path):
            img = Image.open(icon_path).resize((70, 70), Image.Resampling.LANCZOS)
            self.bird_icon = ImageTk.PhotoImage(img)
            canvas.create_image(button_size//2, button_size//2, image=self.bird_icon)
        
        canvas.bind("<Button-1>", lambda e: self.select_and_process_file())
        canvas.bind("<Enter>", lambda e: canvas.itemconfig(self.circle, fill="#c0392b"))
        canvas.bind("<Leave>", lambda e: canvas.itemconfig(self.circle, fill="#e74c3c"))
        canvas.configure(cursor="hand2")
        return canvas
    
    def select_and_process_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file âm thanh",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            filename = os.path.basename(file_path)
            self.status_label.config(text=f"Đang phân tích: {filename}")
            self.result_label.config(text="Đang xử lý...", fg="#f39c12")
            self.root.update_idletasks()
            self.root.after(100, lambda: self.process_file(file_path))
    
    def process_file(self, file_path):
        name, conf = self.classify_audio(file_path)
        result = f"{name}\n(Độ tin cậy: {conf:.1f}%)"
        self.display_result(result, name)

    def display_result(self, text, class_name=None):
        if class_name and class_name not in ["Model chưa tải", "Lỗi xử lý âm thanh", "Lỗi phân loại"]:
            image_path = os.path.join(os.path.dirname(__file__), "class_images", f"{class_name}.jpg")
            if os.path.exists(image_path):
                try:
                    bird_img = Image.open(image_path)
                    bird_img = bird_img.resize((300, 200), Image.Resampling.LANCZOS)
                    self.current_bird_image = ImageTk.PhotoImage(bird_img)
                    self.image_label.config(image=self.current_bird_image)
                except Exception as e:
                    print(f"Lỗi tải ảnh: {e}")
                    self.image_label.config(image="", text="Không tìm thấy ảnh")
            else:
                self.image_label.config(image="", text="Không có ảnh")
        else:
            self.image_label.config(image="", text="")
        
        self.result_label.config(text=text, fg="#27ae60", font=("Arial", 16, "bold"))
        self.status_label.config(text="Hoàn thành", fg="#27ae60")


def main():
    root = tk.Tk()
    app = BirdClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()