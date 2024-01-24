import cv2

# 替换为实际的视频文件路径
video_path = '/home/ustc/Videos/1-24gaitgood-1.mp4'  

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("错误：无法打开视频。")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 计算每0.5秒的帧数间隔
frame_interval = int(fps * 0.5)

# 初始化帧计数器
frame_counter = 0

# 初始化输出帧的列表
saved_frames = []

# 逐帧读取视频
while True:
    ret, frame = cap.read()
    
    # 如果没有更多帧，则退出循环
    if not ret:
        break
    
    # 每0.5秒保存一帧
    if frame_counter % frame_interval == 0:
        # 构造图片文件名
        frame_filename = f'/home/ustc/Videos/pic/1-24gaitgood/frame_{frame_counter//frame_interval}.png'
        # 保存当前帧为图片
        cv2.imwrite(frame_filename, frame)
        saved_frames.append(frame_filename)
        print(f"保存：{frame_filename}")
    
    # 帧计数器增加
    frame_counter += 1

# 释放视频捕 &#8203;``【oaicite:0】``&#8203;
