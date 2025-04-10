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

# --- Загрузка модели и процессора ---
# Укажите путь к локально скачанным файлам модели
# Убедитесь, что эта директория существует и содержит все необходимые файлы
local_model_path = "./wav2vec2-local"

# Загрузка модели и процессора из локального пути
try:
    processor = Wav2Vec2Processor.from_pretrained(local_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(local_model_path)
except OSError as e:
    st.error(f"Ошибка загрузки модели из '{local_model_path}': {e}\n"
             f"Убедитесь, что путь указан верно и файлы модели скачаны.")
    st.stop() # Останавливаем выполнение скрипта, если модель не загружена

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

# Функция распознавания речи из аудиофайла или тензора
def transcribe(audio_path=None, waveform=None, sampling_rate=None):
    if audio_path:
        speech_array, sampling_rate = torchaudio.load(audio_path)
    elif waveform is not None and sampling_rate is not None:
        speech_array = waveform
    else:
        raise ValueError("Either audio_path or waveform and sampling_rate must be provided")

    # Убедимся, что у нас 1 канал (моно)
    if speech_array.ndim > 1 and speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

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
def visualize_audio(audio_path=None, y=None, sr=None):
    if audio_path:
        y, sr = librosa.load(audio_path, sr=16000) # Загружаем и ресемплируем до 16кГц
    elif y is None or sr is None:
         raise ValueError("Either audio_path or y and sr must be provided")
    elif sr != 16000:
        # Если переданы данные, но не с той частотой дискретизации, ресемплируем
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title("Waveform")
    # Используем librosa.stft непосредственно
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
    axs[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axs[1]) # Убедимся, что colorbar привязан к правильной оси
    plt.tight_layout() # Добавим для лучшего расположения
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
    # Читаем байты файла
    audio_bytes = uploaded_file.getvalue()
    # Загружаем аудио тензор из байтов
    waveform, sr = torchaudio.load(BytesIO(audio_bytes))

    # Передаем тензор и частоту дискретизации в transcribe
    predicted = transcribe(waveform=waveform, sampling_rate=sr)
    result = compare_with_target(predicted, target_word)
    report = generate_report(result)

    # Отображаем аудио из исходных байтов
    st.audio(audio_bytes, format="audio/wav")
    st.write(report)
    # Визуализация из загруженных данных (конвертируем тензор в numpy)
    st.pyplot(visualize_audio(y=waveform.numpy().squeeze(), sr=sr))
    st.markdown(get_download_link(report), unsafe_allow_html=True)

# Включение записи с микрофона (streamlit-webrtc)
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

from streamlit_webrtc import WebRtcMode

ctx = webrtc_streamer(
    key='mic_stream_',
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# Логика обработки после остановки записи
if ctx.audio_processor:
    if not ctx.state.playing and len(ctx.audio_processor.frames) > 0:
        st.info("Обработка записи...")
        frames = ctx.audio_processor.frames
        # Объединяем фреймы в один numpy массив
        samples = np.concatenate([f.to_ndarray()[0] for f in frames]).astype(np.float32)
        # Убедимся, что массив одномерный (моно)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1) # Или samples = samples[:, 0] если знаем, что звук в первом канале

        # Конвертируем numpy в torch тензор для транскрипции
        # Добавляем измерение для батча/канала, если нужно torchaudio
        waveform = torch.tensor(samples).unsqueeze(0)
        sr = 16000 # Частота дискретизации WebRTC обычно 16000 или 48000, но Wav2Vec требует 16000

        # Транскрибируем из тензора
        predicted = transcribe(waveform=waveform, sampling_rate=sr)
        result = compare_with_target(predicted, target_word)
        report = generate_report(result)

        # Сохраняем аудио в байтовый буфер для st.audio
        buffer = BytesIO()
        torchaudio.save(buffer, waveform, sr, format="wav")
        buffer.seek(0)

        # Отображаем результаты
        st.audio(buffer, format="audio/wav")
        st.write(report)
        # Визуализируем из numpy массива
        st.pyplot(visualize_audio(y=samples, sr=sr))
        st.markdown(get_download_link(report), unsafe_allow_html=True)

        # Очищаем буфер фреймов для следующей записи
        ctx.audio_processor.frames = []
        st.success("Обработка завершена!")

elif ctx.state.playing:
    # Можно добавить индикатор записи, если нужно
    st.write("Идет запись... Нажмите 'Stop' для завершения.")

# Удаление старого блока обработки во время записи
# if ctx and ctx.state.playing and ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
#     frames = ctx.audio_processor.frames
#     samples = np.concatenate([f.to_ndarray()[0] for f in frames]).astype(np.float32)
#     audio_path = "mic_recording.wav"
#     # Нужно убедиться, что формат и частота дискретизации соответствуют ожиданиям модели
#     # Wav2Vec ожидает 16000 Гц, моно
#     # av фреймы могут иметь другую частоту, проверим первый фрейм
#     sample_rate_mic = frames[0].sample_rate
#     waveform_mic = torch.tensor([samples])
#     if sample_rate_mic != 16000:
#         resampler_mic = T.Resample(orig_freq=sample_rate_mic, new_freq=16000)
#         waveform_mic = resampler_mic(waveform_mic)
#     # Сохраняем ресемплированное аудио
#     torchaudio.save(audio_path, waveform_mic, 16000)
#
#     predicted = transcribe(audio_path=audio_path)
#     result = compare_with_target(predicted, target_word)
#     report = generate_report(result)
#
#     st.audio(audio_path, format="audio/wav")
#     st.write(report)
#     st.pyplot(visualize_audio(audio_path=audio_path))
#     st.markdown(get_download_link(report), unsafe_allow_html=True)
