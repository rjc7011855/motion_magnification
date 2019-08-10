import cv2
import numpy as np
import scipy.signal as signal  # 导入信号滤波类
import scipy.fftpack as fftpack  # 导入对时域信号进行快速傅里叶变换的类
import matplotlib.pyplot as plt
import math

# convert RBG to YIQ
def rgb2ntsc(src):
    [rows, cols] = src.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = np.dot(T, src[i, j])
    return dst


# convert YIQ to RBG
def ntsc2rgb(src):
    [rows, cols] = src.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = np.dot(T, src[i, j])
    return dst


# Build Gaussian Pyramid
def build_gaussian_pyramid(src, level=3):  # src输入图像
    s = src.copy()
    pyramid = [s]
    for i in range(level):  # 0 1 2
        s = cv2.pyrDown(s)  # 先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        pyramid.append(s)
    return pyramid  # 返回高斯金字塔，4维，每一层就是一张图片


# Build Laplacian Pyramid
def build_laplacian_pyramid(src, levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid = []
    for i in range(levels, 0, -1):  # 3 2 1
        GE = cv2.pyrUp(gaussianPyramid[i])  # 先对图像进行升采样（将图像尺寸行和列方向增大一倍），然后再进行高斯平滑；
        L = cv2.subtract(gaussianPyramid[i - 1], GE)  # 高斯金字塔倒数第二次与倒数第一层进行上采样的相减
        pyramid.append(L)
    return pyramid  # 返回拉普拉斯金字塔


# load video from file
def load_video(video_filename):
    cap = cv2.VideoCapture(video_filename)  # 读入视频文件
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
    x = 0
    while cap.isOpened():  # 检查视频文件是否成功打开或初始化摄像头是否成功
        ret, frame = cap.read()  # 按桢读取视频 ret=true 读取桢正确 fluse 失败  frame 每一帧图片
        if ret is True:
            video_tensor[x] = rgb2ntsc(frame)
            # video_tensor[x]=frame
            x += 1
        else:
            break
    return video_tensor, fps


# apply temporal ideal bandpass filter to gaussian video  #bandpass filter 带通滤波器
# 其中signal.fftpack.fft函数中有一个参数为axis，这个参数可以指定矩阵求fft的方向，
# 这里将其指定为0便可以沿着时间方向进行fft计算
def temporal_ideal_filter(tensor, low, high, fps, axis=0):  # 理想带通滤波器滤波
    fft = fftpack.fft(tensor, axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)  # d 一桢多少秒
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


# build gaussian pyramid for video
def gaussian_video(video_tensor, levels=3):
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_gaussian_pyramid(frame, level=levels)
        gaussian_frame = pyr[-1]
        if i == 0:
            vid_data = np.zeros((video_tensor.shape[0], gaussian_frame.shape[0], gaussian_frame.shape[1], 3))
        vid_data[i] = gaussian_frame  # 对每一帧高斯金字塔最后一张图片初始化
    return vid_data  # 返回一个矩阵 每一帧高斯金字塔最后一张图片


# amplify the video
def amplify_video(gaussian_vid, amplification=50):
    return gaussian_vid * amplification


# reconstract video from original video and gaussian video
# 重建的时候，把高斯金字塔最小的一级(已经与放大因子相乘)进行上采样，最后与原视频进行叠加
def reconstract_video(amp_video, origin_video, levels=3):
    final_video = np.zeros(origin_video.shape)  # origin_video=t=video_tensor
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):  # 0 1 2
            img = cv2.pyrUp(img)
        img = img + origin_video[i]
        final_video[i] = img
    return final_video


# save video to files
def save_video(video_tensor, name_):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("./video/name_" + name_ + ".avi", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(ntsc2rgb(video_tensor[i])))
        # writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()


# magnify color                                          #新加色域转换
def magnify_color(video_name, low, high, levels=3, amplification=20):
    t, f = load_video(video_name)  # return video_tensor,fps
    # numbers=t.shape[0]
    # tt=np.zeros((numbers,t.shape[1],t.shape[2],t.shape[3]),dtype=np.float32)
    # for i in range(numbers):
    #    ttt=rgb2ntsc(t[i])
    #    tt[i]=ttt                     #新加的色域转换
    gau_video = gaussian_video(t, levels=levels)
    filtered_tensor = temporal_ideal_filter(gau_video, low, high, f)
    amplified_video = amplify_video(filtered_tensor, amplification=amplification)
    final = reconstract_video(amplified_video, t, levels=3)
    # number_two=final.shape[0]
    # final_image=np.zeros((number_two,final.shape[1],final.shape[2],final.shape[3]),dtype=np.float32)
    # for i in range(number_two):
    #    final_two=ntsc2rgb(final[i])
    #    final_image[i]=final_two
    save_video(final)


# build laplacian pyramid for video
def laplacian_video(video_tensor, levels=3):
    tensor_list = []
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_laplacian_pyramid(frame, levels=levels)  # 对每帧图片构建拉普拉斯金字塔
        if i == 0:
            for k in range(levels):  # 0 1 2对金字塔每张图片进行初始化
                tensor_list.append(np.zeros((video_tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)))
        for n in range(levels):  # 0 1 2
            tensor_list[n][i] = pyr[n]  # 4维拉普拉斯金字塔 3行i列的矩阵
    return tensor_list  # 返回3行i列的矩阵，每个位置存放一张图片


# butterworth bandpass filter  巴特沃兹带通滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):  # 对每张图片运用巴特沃兹带通滤波器
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y


# reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list, levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)  # 最后一个元素的shape
    print('filter_tensor_list[-1].shape', filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]  # 第0行
        for n in range(levels - 1):  # 0 1
            up = cv2.pyrUp(up) + filter_tensor_list[n + 1][i]  # 可以改为up=cv2.pyrUp(up)
        final[i] = up
    return final

def nonlinear(sums,amplication):
    amplications=(math.exp(-sums+2)+1)*amplication
    return amplications

# manify motion
def magnify_motion(video_name, low, high, levels=3,weight1=3,weight2=2,weight3=1,amplification=10):
    t, f = load_video(video_name)
    numbers = t.shape[0]
    lap_video_list = laplacian_video(t, levels=levels)  ##返回3行i列的矩阵
    filter_tensor_list = []
    butter_bandpass_filter_list = []
    sum_array = np.zeros((levels, numbers), dtype=np.float64)
    sums_array = np.zeros(numbers,dtype=np.float64)
    amplification_array = np.zeros(numbers, dtype=np.float64)
    sums=0
    for i in range(levels):
        filter_tensor = butter_bandpass_filter(lap_video_list[i], low, high, f)  # f =fs
        print("lap_video_list[i].shape", lap_video_list[i].shape)
        print('filter_tensor.shape', filter_tensor.shape)
        # print('filter_tensor[0]',filter_tensor[0])
        new_filter_tensor = np.zeros((numbers, filter_tensor.shape[1], filter_tensor.shape[2], filter_tensor.shape[3]),
                                     dtype=np.float64)
        # new_filter_tensor=[]
        for j in range(numbers):
            sum_=np.sum(np.abs(filter_tensor[j]))
            sum_array[i, j] = sum_
            if i==levels-1:
                sums=weight1*16*sum_array[0,j]+weight2*4*sum_array[1,j]+weight3*sum_array[2,j]
                sums_array[j]=sums
            print('sums',sums)
            amplifications=nonlinear(sums,amplification)
            amplification_array[j]=amplifications
            new_filter_tensor[j] = filter_tensor[j] * amplifications
            print('is i: ', i, ' j: ', j)
        butter_bandpass_filter_list.append(filter_tensor)
        filter_tensor_list.append(new_filter_tensor)
    sum_1=sum_array[0,:]
    #print('sum_1',sum_1)
    sum_2=sum_array[1,:]
    #print('sum_2',sum_2)
    sum_3=sum_array[2,:]
    #print('sum_3',sum_3)
  #  plt.figure(1)
  #  plt.plot(sum_1, 'b-.')
   # plt.legend(["sum_1"])
   # plt.title("sum_1 " )
   # plt.grid()
   # plt.savefig('./figure/sum_1.png')
    #plt.show()
   # plt.figure(2)
   # plt.plot(sum_2, 'b-.')
   # plt.legend(["sum_2"])
    #plt.title("sum_2 " )
   # plt.grid()
    #plt.savefig('./figure/sum_2.png')
    #plt.show()
    #plt.figure(3)
    #plt.plot(sum_3, 'b-.')
    #plt.legend(["sum_3"])
    #plt.title("sum_3 " )
    #plt.grid()
    #plt.savefig('./figure/sum_3.png')
    #plt.show()
    plt.figure(1)
    plt.plot(sums_array, 'b-.')
    plt.legend(["sums"])
    plt.title("sums " )
    plt.grid()
    plt.savefig('./figure/sums.png')
    plt.show()
    plt.figure(2)
    plt.plot(amplification_array, 'b-.')
    plt.legend(["amplications"])
    plt.title("amplications")
    plt.grid()
    plt.savefig('./figure/amplications.png')
    plt.show()
    recon = reconstract_from_tensorlist(filter_tensor_list)
    butter = reconstract_from_tensorlist(butter_bandpass_filter_list)
    final = t + recon
    save_video(video_tensor=final, name_="final")
    save_video(video_tensor=recon, name_="recon")
    # save_video(video_tensor=butter,name_="butter")


if __name__ == "__main__":
    # magnify_color("face.mp4",0.4,3)
    # magnify_motion("guitar.mp4",0.4,3)   #origial data
    magnify_motion("baby.mp4", 0.498, 0.698)
