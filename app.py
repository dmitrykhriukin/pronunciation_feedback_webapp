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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av

# (оставшийся код опущен здесь для краткости — он будет полностью вставлен в файл)
