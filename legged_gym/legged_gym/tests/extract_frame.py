import cv2

# 替换为实际的视频文件路径
video_path = '/home/ustc/Videos/1-29parkour.mp4'

# 指定开始和结束时间（以秒为单位）
start_time = 17  # 从视频的第10秒开始
end_time = 26    # 到视频的第60秒结束

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("错误：无法打开视频。")
    exit()

# 获取视频的帧率和总帧数
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 计算开始和结束时间对应的帧编号
start_frame = int(fps * start_time)
end_frame = int(fps * end_time)
end_frame = min(end_frame, total_frames)  # 防止结束帧超出视频总帧数

# 计算每0.5秒的帧数间隔
frame_interval = int(fps * 0.1)

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
    
    # 只处理位于指定时间范围内的帧
    if start_frame <= frame_counter <= end_frame:
        # 每0.5秒保存一帧
        if frame_counter % frame_interval == 0:
            # 构造图片文件名
            frame_filename = f'/home/ustc/Videos/pic/1-29parkour_big/frame_{frame_counter//frame_interval}.png'
            # 保存当前帧为图片
            cv2.imwrite(frame_filename, frame)
            saved_frames.append(frame_filename)
            print(f"保存：{frame_filename}")
    
    # 帧计数器增加
    frame_counter += 1

# 释放视频捕获对象
cap.release()
