import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import queue

# 参数设置
Fs = 44100  # 采样频率
timeLength = 0.5  # 缩短采样时长为 0.02 秒
frameSize = int(Fs * timeLength)  # 计算每帧采样点数，对应 0.02 秒
xdata = (np.arange(0, frameSize // 2)) * (Fs / frameSize)  # 频率数据
timeAxis = np.arange(0, frameSize) / Fs  # 时域时间轴

# 创建图形窗口
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle('Real-time Spectrum')

# 初始化时域和频域图
audioIn = np.zeros(frameSize)  # 初始化音频数据
line1, = ax1.plot(timeAxis, audioIn)  # 时域图
line2, = ax2.plot(xdata, np.zeros_like(xdata), 'r')  # 频域图

# 配置时域轴属性 (0 到 0.02 秒)
ax1.set_xlim(0, timeLength)
ax1.set_ylim(-0.1, 0.1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')

# 配置频域轴属性 (频率 0 到 10000 Hz)
ax2.set_xlim(0, 10000)  # 频率轴修改为 0 到 10000 Hz
ax2.set_ylim(0, 6)
ax2.set_xscale('linear')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude (log scale)')

# 用于存储音频数据的队列
audio_queue = queue.Queue()

# 音频回调函数
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata[:, 0])  # 将音频数据存入队列

# 绘图更新函数
def update_plot():
    while not audio_queue.empty():
        frame = audio_queue.get()  # 从队列中获取音频数据
        ydata_fft = fft(frame)  # 傅里叶变换
        ydata_abs = np.abs(ydata_fft[:frameSize // 2])  # 取绝对值

        # 更新时域图
        line1.set_ydata(frame)

        # 更新频谱图，频率限制在 0 到 10000 Hz
        freq_limit = np.where(xdata <= 10000)[0][-1]  # 找到 10000 Hz 的索引位置
        line2.set_xdata(xdata[:freq_limit + 1])  # 限制频率显示到 10000 Hz
        line2.set_ydata(np.log(1 + ydata_abs[:freq_limit + 1]))  # 更新频谱图

        # 动态调整时域图纵轴范围
        ax1.set_ylim(min(frame), max(frame))

    plt.pause(0.05)  # 每次暂停一段时间

# 实时采集和显示
with sd.InputStream(channels=1, samplerate=Fs, blocksize=frameSize, callback=audio_callback, latency='high'):
    while True:
        update_plot()  # 定期更新绘图