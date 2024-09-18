import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

# 录音参数
fs = 11025  # 采样率
duration = 5  # 录制5秒

# 录制音频
print("开始录音...")
MyRecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
sd.wait()  # 等待录音完成
print("录音结束")

# 播放录音
sd.play(MyRecording, fs)
sd.wait()

# 绘制原始信号的时域波形
plt.subplot(3, 2, 1)
plt.plot(MyRecording[:, 0])
plt.title('Original Time Domain - Left Channel')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')

# 计算原始信号的傅里叶变换（频域分析）
N = len(MyRecording)
Y = fft(MyRecording[:, 0])
f = np.linspace(0, fs, N)
P = np.abs(Y) / N

# 绘制原始信号的频域
plt.subplot(3, 2, 2)
plt.plot(f[:N // 2], P[:N // 2])
plt.title('Original Frequency Domain - Left Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# 增加均匀噪声
noise_amplitude = 0.005
noise = noise_amplitude * np.random.randn(*MyRecording.shape)
NoisyRecording = MyRecording + noise

# 绘制带噪声信号的时域波形
plt.subplot(3, 2, 3)
plt.plot(NoisyRecording[:, 0])
plt.title('Noisy Time Domain - Left Channel')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')

# 计算带噪声信号的傅里叶变换（频域分析）
Y_noisy = fft(NoisyRecording[:, 0])
P_noisy = np.abs(Y_noisy) / N

# 绘制带噪声信号的频域
plt.subplot(3, 2, 4)
plt.plot(f[:N // 2], P_noisy[:N // 2])
plt.title('Noisy Frequency Domain - Left Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# 使用带通滤波器滤除噪声（保留300Hz-3400Hz的频率）
fc_low = 300
fc_high = 3000

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

FilteredRecording = np.apply_along_axis(bandpass_filter, 0, NoisyRecording, fc_low, fc_high, fs)

# 绘制滤波后信号的时域波形
plt.subplot(3, 2, 5)
plt.plot(FilteredRecording[:, 0])
plt.title('Filtered Time Domain - Left Channel')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')

# 计算滤波后信号的傅里叶变换（频域分析）
Y_filtered = fft(FilteredRecording[:, 0])
P_filtered = np.abs(Y_filtered) / N

# 绘制滤波后信号的频域
plt.subplot(3, 2, 6)
plt.plot(f[:N // 2], P_filtered[:N // 2])
plt.title('Filtered Frequency Domain - Left Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# 保存滤波后的音频文件
write('filtered_myspeech.wav', fs, FilteredRecording)

plt.tight_layout()
plt.show()