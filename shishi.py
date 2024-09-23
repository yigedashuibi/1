import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 参数设置
Fs = 44100  # 采样频率
timeLength = 0.02  # 缩短采样时长为 0.02 秒
frameSize = int(Fs * timeLength)  # 每帧采样点数，基于新的时间长度
xdata = (np.arange(0, frameSize // 2)) * (Fs / frameSize)  # 频率数据
timeAxis = np.arange(0, frameSize) / Fs  # 时域时间轴

# 创建图形窗口
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle('Real-time Spectrum')

# 初始化时域和频域图
audioIn = np.zeros(frameSize)  # 初始化音频数据
line1, = ax1.plot(timeAxis, audioIn)  # 时域图
line2, = ax2.plot(xdata, np.zeros_like(xdata), 'r')  # 频域图

# 配置轴属性
ax1.set_xlim(0, timeLength)
ax1.set_ylim(-0.01, 0.01)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')

# 将频率轴的最大值设为 10,000 Hz
ax2.set_xlim(0, 10000)
ax2.set_ylim(0, 6)
ax2.set_xscale('linear')  # 取消对数轴，改为线性轴以匹配新的频率范围
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude (log scale)')

# 更新绘图函数
def update_plot(frame):
    ydata_fft = fft(frame)  # 傅里叶变换
    ydata_abs = np.abs(ydata_fft[:frameSize // 2])  # 取绝对值
    line1.set_ydata(frame)  # 更新时域图
    line2.set_ydata(np.log(1 + ydata_abs))  # 更新频谱图
    ax1.set_ylim(min(frame), max(frame))  # 动态调整时域图纵轴范围
    fig.canvas.draw()
    fig.canvas.flush_events()

# 音频回调函数
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    update_plot(indata[:, 0])

# 实时采集和显示
with sd.InputStream(channels=1, samplerate=Fs, blocksize=frameSize, callback=audio_callback):
    plt.show()