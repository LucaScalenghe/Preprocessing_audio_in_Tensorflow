import tensorflow as tf
import tensorflow_io as tfio


LABELS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


def get_audio_and_label(filename):
    audio_binary = tf.io.read_file(filename)
    audio, sampling_rate = tf.audio.decode_wav(audio_binary)

    path_parts = tf.strings.split(filename, '/')
    path_end = path_parts[-1]
    file_parts = tf.strings.split(path_end, '_')
    label = file_parts[0]

    audio = tf.squeeze(audio)

    zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
    audio_padded = tf.concat([audio, zero_padding], axis=0)

    return audio_padded, sampling_rate, label
    

def get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio_padded, sampling_rate, label = get_audio_and_label(filename)
    
    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate) 

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram, sampling_rate, label


def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s=0.04, frame_step_in_s=0.02, num_mel_bins=40, lower_frequency=20, upper_frequency=4000):
    spectrogram, sampling_rate, label = get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s)
    num_spectrogram_bins = spectrogram.shape[1]
    
    #Search what this function does
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sampling_rate,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    mel_spectrogram.shape
    return log_mel_spectrogram, label


def get_mfccs(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients=10):
    #First you obtain the log_mel_spectrogram
    log_mel_spectrogram, label = get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    
    #Reshapping the mfccs in order to have only the desired coefficients
    mfccs = mfccs[..., :num_coefficients]
    return mfccs, label
