Hi! Do you know djent?
ChatGPT

Yes, I am familiar with djent. Djent is a subgenre of progressive metal that originated in the mid-2000s. It is characterized by heavy, palm-muted, low-pitched guitar riffs with complex polyrhythms and syncopated patterns, often achieved using extended-range guitars and seven- or eight-string instruments. The term "djent" was coined as an onomatopoeia for the distinctive sound of the palm-muted guitar chugs. The genre is closely associated with bands such as Meshuggah, Animals as Leaders, Periphery, and Tesseract.
User
Right. So how would you go about extracting those palm-muted guitar chucks as morse code (dots, underscores, spaces). Is there an AI tool that could help me with that?
ChatGPT

Extracting palm-muted guitar chugs as Morse code using an AI tool would be a challenging task since the sound of a guitar chug can be quite complex, and it may be difficult for an AI to distinguish between individual chugs accurately. However, there are some audio analysis tools available that could potentially help you achieve this.

One such tool is Sonic Visualizer, an open-source audio analysis software that allows you to visualize and analyze audio files in various ways, including spectrograms, waveforms, and note views. With Sonic Visualizer, you can visualize the waveform of the guitar chugs and then use the software's built-in annotation tools to manually transcribe the chugs as Morse code dots, underscores, and spaces.

Another option is to use a machine learning-based audio analysis tool, such as Google's AudioSet or the open-source TensorFlow audio recognition library. These tools are designed to recognize and classify audio signals based on their characteristics and could potentially be trained to recognize the specific sound of palm-muted guitar chugs and transcribe them as Morse code.

However, developing such a system would require a significant amount of audio data to train the model accurately and may still require manual intervention to ensure accuracy.
User
That's a good start! Maybe an audio filter would help before feeding the music into the AI. To only keep the particular frequencies that make up the chugs. Do you know a library that could help me with that? All steps should all be scriptable, from taking in the audio source, filtering and then the actual AI-aided process of tranlating the chugs into morse.
ChatGPT

Yes, using an audio filter to isolate the frequencies of the guitar chugs before feeding the audio into an AI model could potentially improve the accuracy of the transcription. One popular Python library for audio processing and filtering is Librosa, which has a wide range of functions for analyzing and manipulating audio signals.

You can use Librosa to load the audio file and apply a bandpass filter to isolate the frequencies of the guitar chugs. Here's some sample Python code that demonstrates how to load an audio file and apply a bandpass filter using Librosa:

python

import librosa
import numpy as np

# Load the audio file
y, sr = librosa.load('audio_file.wav')

# Set the bandpass filter parameters
fmin = 100 # Minimum frequency to keep
fmax = 500 # Maximum frequency to keep
n_fft = 2048 # Number of FFT bins
hop_length = 512 # Number of samples between FFT windows

# Apply the bandpass filter
y_filt = librosa.effects.bandpass(y, fmin, fmax, n_fft=n_fft, hop_length=hop_length)

# Compute the spectrogram of the filtered signal
S = np.abs(librosa.stft(y_filt, n_fft=n_fft, hop_length=hop_length))

This code loads the audio file audio_file.wav using Librosa, applies a bandpass filter to isolate frequencies between 100 Hz and 500 Hz, and then computes the spectrogram of the filtered signal. You can then use the spectrogram as input to an AI model for transcription.

For the AI transcription part, there are various libraries and frameworks you can use, such as TensorFlow or PyTorch, to train and deploy a machine learning model to transcribe the guitar chugs into Morse code.
User
That sounds great! How would I exchange the WAV input for let's say a Spotify stream?
ChatGPT

Extracting audio from a Spotify stream is not straightforward since Spotify uses digital rights management (DRM) to prevent unauthorized access to their music content. However, if you have access to the audio source in a digital format, you can convert it to a suitable format and feed it to the processing pipeline.

One way to access the audio content of Spotify is through the Spotify Web API, which allows developers to access Spotify's music catalog and retrieve information about tracks, albums, and artists. You can use the Spotify Web API to retrieve information about a specific track, including its Spotify URI, which you can then use to access the track's audio content using a third-party tool or library.

There are several third-party tools available that can help you download the audio content of a Spotify track, such as SpotDL or Spotipy. These tools allow you to download a track's audio content in a variety of formats, including MP3, WAV, and FLAC. Once you have the audio content in a suitable format, you can then feed it into the processing pipeline, as described in the previous answer.

It's worth noting that downloading music from Spotify without authorization is against their terms of service and may be illegal in some jurisdictions. Therefore, make sure you have the necessary rights and permissions before downloading or using any copyrighted music content.
User
Then I'd probably avoid downloading and just let my model listen to a live stream via computer audio. Once it's set up, it should be able to transcribe on the go, right? So how would I take in computer audio in the script above?
ChatGPT

Yes, you can certainly use live audio input from your computer to transcribe the guitar chugs in real-time. To capture audio input from your computer's audio device, you can use a Python library called PyAudio.

Here's some sample Python code that demonstrates how to use PyAudio to capture audio input from your computer's audio device and apply a bandpass filter using Librosa:

python

import pyaudio
import librosa
import numpy as np

# Set the audio capture parameters
RATE = 44100 # Sample rate
CHUNKSIZE = 1024 # Buffer size

# Set the bandpass filter parameters
fmin = 100 # Minimum frequency to keep
fmax = 500 # Maximum frequency to keep
n_fft = 2048 # Number of FFT bins
hop_length = 512 # Number of samples between FFT windows

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream to capture audio input
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNKSIZE)

# Continuously capture and process audio input
while True:
    # Read a chunk of audio input from the stream
    data = stream.read(CHUNKSIZE)
    
    # Convert the audio data to a NumPy array
    y = np.frombuffer(data, dtype=np.float32)
    
    # Apply the bandpass filter
    y_filt = librosa.effects.bandpass(y, fmin, fmax, n_fft=n_fft, hop_length=hop_length)
    
    # Compute the spectrogram of the filtered signal
    S = np.abs(librosa.stft(y_filt, n_fft=n_fft, hop_length=hop_length))
    
    # Perform Morse code transcription using an AI model
    # (code for this step depends on the specific AI model used)

This code initializes PyAudio, opens a stream to capture audio input from your computer's audio device, and continuously reads and processes audio input in a while loop. The audio input is processed by applying a bandpass filter using Librosa, and the resulting spectrogram can be used for Morse code transcription using an AI model.

You'll need to install PyAudio and its dependencies before running this code. Note that the specific parameters used for the bandpass filter and spectrogram computation may need to be adjusted depending on the characteristics of the guitar chugs you're trying to transcribe.
User
Oh that's a good point. The chugs could vary from song to song, so we'd need a AI-aided process first to guess the frequency range of the chugs in a song, like from a  30sec-excerpt and set the filter accordingly before performing the actual analysis. How would I do that?
ChatGPT

To automatically estimate the frequency range of the guitar chugs in a song, you can use a technique called onset detection, which identifies the points in time where musical events (such as guitar chugs) begin in a song. By analyzing the frequencies present in the onset events, you can estimate the frequency range of the guitar chugs and set the bandpass filter accordingly.

Here's some sample Python code that demonstrates how to use Librosa to perform onset detection on a 30-second excerpt of a song and estimate the frequency range of the guitar chugs:

python

import librosa
import numpy as np

# Load a 30-second excerpt of a song
filename = 'example_song.wav'
y, sr = librosa.load(filename, duration=30)

# Compute the onset envelope of the audio signal
onset_env = librosa.onset.onset_strength(y, sr=sr)

# Compute the onset times of the audio signal
onset_times = librosa.onset.onset_detect(y, sr=sr, onset_envelope=onset_env)

# Compute the chromagram of the audio signal
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Compute the mean chroma for each onset event
mean_chroma = []
for i in range(len(onset_times)):
    onset_frame = librosa.time_to_frames(onset_times[i], sr=sr)
    chroma_slice = chroma[:, onset_frame:onset_frame+2048]
    mean_chroma.append(np.mean(chroma_slice, axis=1))

# Compute the frequency range of the guitar chugs
mean_chroma = np.array(mean_chroma)
frequencies = librosa.cqt_frequencies(12 * 3 + 1, fmin=librosa.note_to_hz('C1'))
guitar_chugs = mean_chroma[:, 45:90]
fmin = np.min(guitar_chugs.dot(frequencies[45:90])) - 50
fmax = np.max(guitar_chugs.dot(frequencies[45:90])) + 50

# Set the bandpass filter parameters
n_fft = 2048 # Number of FFT bins
hop_length = 512 # Number of samples between FFT windows

# Initialize a PyAudio stream to capture live audio input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=n_fft)

# Continuously capture and process audio input
while True:
    # Read a chunk of audio input from the stream
    data = stream.read(n_fft)
    
    # Convert the audio data to a NumPy array
    y = np.frombuffer(data, dtype=np.float32)
    
    # Apply the bandpass filter
    y_filt = librosa.effects.bandpass(y, fmin, fmax, n_fft=n_fft, hop_length=hop_length)
    
    # Compute the spectrogram of the filtered signal
    S = np.abs(librosa.stft(y_filt, n_fft=n_fft, hop_length=hop_length))
    
    # Perform Morse code transcription using an AI model
    # (code for this step depends on the specific AI model used)

This code loads a 30-second excerpt of a song and performs onset detection to identify the points in time where musical events begin. For each onset event, the code computes the mean chroma of the audio signal in a small time window around the onset time. The resulting mean chroma values are then used to estimate the frequency range of the guitar chugs by finding the minimum and maximum frequencies present in the chroma values corresponding to the guitar notes.

The estimated frequency range


