# Install necessary libraries
!pip install transformers
!pip install datasets
!pip install torchaudio

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
import torch
from google.colab import files

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform.squeeze().numpy(), sample_rate

def transcribe_audio(audio_file):
    model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"

    # Load pre-trained model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name_or_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path)

    # Load and preprocess the audio file
    waveform, sample_rate = load_audio(audio_file)

    # Ensure the sample rate matches the model's expected rate
    desired_sample_rate = 16000
    if sample_rate != desired_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
        waveform = resampler(torch.tensor(waveform)).numpy()

    # Prepare the audio for prediction
    inputs = tokenizer(waveform, return_tensors="pt", padding="longest")

    # Perform inference
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # Decode the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)

    return transcription[0]

# Upload your audio file
uploaded = files.upload()
audio_file = list(uploaded.keys())[0]

# Transcribe the audio file
transcription = transcribe_audio(audio_file)
print("Transcription:", transcription)
