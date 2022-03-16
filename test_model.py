import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
import numpy as np
import os
import glob
from pathlib import Path

# .eval() is not supported when eager execution is enabled
tf.compat.v1.disable_eager_execution()

print (tf.__version__)

TFLITE_FILE_PATH  = './model7_1.tflite'

# Load tflite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']

# print (input_shape)
# print (output_shape)

# Random inputs for testing the model
# quantized model
# input_data = np.array(np.random.random_integers(low = -128, high=127, size=input_shape), dtype=np.int8)
# #print (input_data)

# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

# float model
# input_data = np.array(np.random.uniform(low = -247.0, high = 30.0, size=input_shape), dtype=np.float32)
# #print (input_data)

# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)


sample_rate = 8000
window_size_ms = 64
window_step_ms = 56
window_size_samples = 512      # window_size_ms / 1000 * sample_rate
window_stride_samples = 448    # window_step_ms / 1000 * sample_rate
fingerprint_width = 13 # feature bin count
filterbank_channel_cnt = 13

# old way of quantizing (TODO: Maybe not needed)
# mfcc_features_min = -247.0
# mfcc_features_max = 30.0

# HUMAN_WAV       = 'dataset/6902-2-0-5.wav'
# INDUSTRIAL_WAV  = 'dataset/518-4-0-0.wav'
# TRAFFIC_WAV     = 'dataset/6988-5-0-0.wav'
# NO_8k = 'data/no8k/0a2b400e_nohash_0.wav'

for filename in glob.glob(os.path.join('data/cat8k', '*.wav')):
    wav_loader = io_ops.read_file(filename)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=sample_rate)

    spectrogram = audio_ops.audio_spectrogram(
            wav_decoder.audio,
            window_size=window_size_samples,
            stride=window_stride_samples,
            magnitude_squared=True)
        
    mfcc = audio_ops.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        upper_frequency_limit=3900.0,
        lower_frequency_limit=125.0,
        filterbank_channel_count=filterbank_channel_cnt,
        dct_coefficient_count=fingerprint_width)

    sess = tf.compat.v1.Session()

    with sess.as_default():
        np_audio = wav_decoder.audio.eval() 
        np_spectrogram = spectrogram.eval()
        np_mfcc = mfcc.eval()


    # Convert mfcc to a format that suits the model
    # For example from (1, 17, 13) to (1, 221)
    np_mfcc_arr = np.expand_dims(np.matrix.flatten(np.squeeze(np_mfcc[0])), axis=0)

    #print (np_mfcc_arr)

    input_scale, input_zero_point = input_details[0]["quantization"]
    # print (input_scale) # 0.067..
    # print (input_zero_point) # -26
    
    np_mfcc_arr = (np_mfcc_arr / input_scale)
    np_mfcc_arr = np.array(np_mfcc_arr + input_zero_point, dtype=np.int8)

    # Old way of quantizing
    # for i, num in enumerate(np_mfcc_arr[0]):
    #     quantized_value = int(round(
    #                 (255 * (num - mfcc_features_min)) / (mfcc_features_max - mfcc_features_min)))
    #     if quantized_value < 0:
    #         quantized_value = 0
    #     if quantized_value > 255:
    #         quantized_value = 255
    #     quantized_value -= 128

    #     np_mfcc_arr[0][i] = quantized_value

    # np_mfcc_arr = np.asarray(np_mfcc_arr, dtype=np.int8)
    #print (np_mfcc_arr)
    # Uncomment for random input
    #np_mfcc_arr = np.random.randint(low=-128, high=127, size=(1, 221), dtype=np.int8)

    interpreter.set_tensor(input_details[0]['index'], np_mfcc_arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_prediction = output_data.argmax()
    print (filename)
    print (output_data)


# REFERENCES:
# https://stackoverflow.com/questions/58786001/tflite-inference-on-video-input
# https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
# https://www.tensorflow.org/lite/guide/inference