# Простой прототип ИИ-системы обратной связи по произношению английских слов
# Используем библиотеку torchaudio и pre-trained модель wav2vec 2.0 + Web-интерфейс на Streamlit

import torchaudio
import torchaudio.transforms as T
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import os
from glob import glob
import pronouncing
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
import streamlit as st
from io import BytesIO
import base64
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# Загрузка модели и процессора
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Загрузка и подготовка Common Voice (английский)
def extract_common_voice(target_words, source_dir, out_dir, max_per_word=10):
    tsv_file = os.path.join(source_dir, "validated.tsv")
    clips_dir = os.path.join(source_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    counters = {w: 0 for w in target_words}

    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row["sentence"].lower().strip()
            for word in target_words:
                if word in sentence.split() and counters[word] < max_per_word:
                    src_audio = os.path.join(clips_dir, row["path"])
                    dst_dir = os.path.join(out_dir, word)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_audio = os.path.join(dst_dir, row["path"])
                    shutil.copy(src_audio, dst_audio)
                    counters[word] += 1
                    break
    print("✅ Извлечение завершено. Файлы сохранены в:", out_dir)

# Функция распознавания речи из аудиофайла
def transcribe(audio_path):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    resampler = T.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze()
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# Получение фонем из слова через CMU Pronouncing Dictionary
def get_phonemes(word):
    phones = pronouncing.phones_for_word(word)
    return phones[0] if phones else None

# Сравнение с эталоном и фонемами + отчёт
def compare_with_target(predicted, target):
    result = {}
    result["prediction"] = predicted
    result["expected"] = target
    result["match"] = predicted == target.lower()
    result["expected_phones"] = get_phonemes(target)
    result["predicted_phones"] = get_phonemes(predicted)
    return result

# Визуализация waveform и спектрограммы

def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title("Waveform")
    D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
    axs[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axs)
    return fig

# Генерация отчета об ошибках

def generate_report(result):
    report_lines = []
    report_lines.append(f"Ожидалось слово: {result['expected']}")
    report_lines.append(f"Произнесено: {result['prediction']}")
    report_lines.append(f"Фонемы (эталон): {result['expected_phones']}")
    report_lines.append(f"Фонемы (ваши): {result['predicted_phones']}")
    match_str = "✅ Совпадает с эталоном" if result['match'] else "❌ Есть отклонения"
    report_lines.append(match_str)
    return "\n".join(report_lines)

# Скачать отчёт как .txt файл

def get_download_link(text, filename="feedback.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Скачать отчёт</a>'
    return href

# Streamlit-интерфейс
st.title("🎙️ Оценка произношения английских слов")
target_word = st.text_input("Введите слово для тренировки:", "photograph")
uploaded_file = st.file_uploader("Загрузите .wav файл:", type=["wav"])

# Использование загруженного файла
if uploaded_file is not None:
    audio_path = f"temp_upload.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    predicted = transcribe(audio_path)
    result = compare_with_target(predicted, target_word)
    report = generate_report(result)

    st.audio(audio_path, format="audio/wav")
    st.write(report)
    st.pyplot(visualize_audio(audio_path))
    st.markdown(get_download_link(report), unsafe_allow_html=True)

# Включение записи с микрофона (streamlit-webrtc)
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

ctx = webrtc_streamer(
    key="mic_stream",
    mode="SENDRECV",
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if ctx and ctx.state.playing and ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
    frames = ctx.audio_processor.frames
    samples = np.concatenate([f.to_ndarray()[0] for f in frames]).astype(np.float32)
    audio_path = "mic_recording.wav"
    torchaudio.save(audio_path, torch.tensor([samples]), 16000)

    predicted = transcribe(audio_path)
    result = compare_with_target(predicted, target_word)
    report = generate_report(result)

    st.audio(audio_path, format="audio/wav")
    st.write(report)
    st.pyplot(visualize_audio(audio_path))
    st.markdown(get_download_link(report), unsafe_allow_html=True)
