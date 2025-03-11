import streamlit as st
import cv2
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pyautogui
import time
from emotion_analyzer import EmotionAnalyzer
import pandas as pd
import os

# Page config
st.set_page_config(layout="wide")

# Custom CSS to reduce padding and margins
st.markdown("""
<style>
    .block-container {padding-top: 3rem; padding-bottom: 0rem;}
    div[data-testid="stVerticalBlock"] > div {padding-top: 0rem; padding-bottom: 0rem;}
    div.stMarkdown p {margin-bottom: 0.5rem;}
    div[data-testid="stHeader"] {padding-top: 0rem;}
    div.stVideo > video {height: 300px !important;}
</style>
""", unsafe_allow_html=True)

# Create layout with tighter spacing
col1, col2 = st.columns([1, 3])

# Left column - participant videos
with col1:
    # 用户1视频源
    img_placeholder_1 = st.empty()
    spectrum_placeholder_1 = st.empty()  # 用户1的情绪光谱条
    
    # 添加分割线
    st.markdown("<hr style='margin: 5px 0px; border: 0; height: 1px; background: #bbb;'>", unsafe_allow_html=True)
    
    # 用户2视频源
    img_placeholder_2 = st.empty()
    spectrum_placeholder_2 = st.empty()  # 用户2的情绪光谱条
    
    # 添加分割线
    st.markdown("<hr style='margin: 5px 0px; border: 0; height: 1px; background: #bbb;'>", unsafe_allow_html=True)
    
    # 用户3视频源
    img_placeholder_3 = st.empty()

# Right column - main video and diagram
with col2:
    try:
        st.video('vid/vid.mp4', start_time=0)
    except Exception as e:
        st.error(f"Error loading video: {e}")
    
    # 创建情绪图表的占位符
    chart_placeholder = st.empty()

# Screen capture settings
class ScreenCapture:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def capture(self):
        # Capture the specified region of the screen
        try:
            screenshot = pyautogui.screenshot(region=(self.x, self.y, self.width, self.height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            st.error(f"Error capturing screen: {e}")
            return np.zeros((150, 200, 3), dtype=np.uint8)  # 减小默认尺寸

# Initialize screen captures
# Adjusted to start from 200px from top with 16:9 aspect ratio
screen1 = ScreenCapture(0, 200, 320, 180)  # User 1 region 
screen2 = ScreenCapture(0, 400, 320, 180)  # User 2 region 
screen3 = ScreenCapture(0, 600, 320, 180)  # Moderator region

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# Add session data storage
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = {
        'timestamp': [],
        'user1_score': [],
        'user2_score': [],
        'emotion_diff': []
    }
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# 现在emotion_analyzer已经初始化，可以安全地使用它
# 创建初始图表
with col2:
    # 创建初始图表
    times = list(range(len(st.session_state.emotion_data['timestamp'])))
    fig = go.Figure()
    if len(times) > 0:  # 只有当有数据时才添加轨迹
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user1_score'], 
                                mode='lines+markers', name='User 1'))
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user2_score'], 
                                mode='lines+markers', name='User 2'))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=180,  # 减小图表高度
        xaxis_title="Time",
        yaxis_title="Emotion Score",
        yaxis=dict(range=[-1, 1])
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

# Add a start/stop button
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button('Start/Stop Capture'):
    st.session_state.running = not st.session_state.running
    
    # When starting, record the start time
    if st.session_state.running:
        st.session_state.start_time = datetime.now()
    
    # When stopping, save the emotion data to CSV
    if not st.session_state.running and len(st.session_state.emotion_data['timestamp']) > 0:
        # Create dataframe from collected data
        df = pd.DataFrame(st.session_state.emotion_data)
        
        # Create directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Save to CSV with timestamp in filename
        filename = f"logs/emotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        st.success(f"Emotion data saved to {filename}")
        
        # Find top 5 moments with highest emotion difference
        df['abs_diff'] = abs(df['user1_score'] - df['user2_score'])
        top_diffs = df.nlargest(5, 'abs_diff')
        
        # Find top 5 moments with lowest emotion difference
        lowest_diffs = df.nsmallest(5, 'abs_diff')
        
        # Display the peak differences
        st.subheader("Top 5 Highest Emotion Difference Moments")
        for i, (_, row) in enumerate(top_diffs.iterrows(), 1):
            st.write(f"Peak {i}: At {row['timestamp']}, User 1: {row['user1_score']:.2f}, "
                    f"User 2: {row['user2_score']:.2f}, Difference: {row['abs_diff']:.2f}")
        
        # Display the lowest differences
        st.subheader("Top 5 Lowest Emotion Difference Moments")
        for i, (_, row) in enumerate(lowest_diffs.iterrows(), 1):
            st.write(f"Moment {i}: At {row['timestamp']}, User 1: {row['user1_score']:.2f}, "
                    f"User 2: {row['user2_score']:.2f}, Difference: {row['abs_diff']:.2f}")
        
        # Reset the data for next session
        st.session_state.emotion_data = {
            'timestamp': [],
            'user1_score': [],
            'user2_score': [],
            'emotion_diff': []
        }
        st.session_state.start_time = None

# Main loop to update video feeds
while st.session_state.running:
    try:
        # 直接使用ScreenCapture对象的capture方法获取视频帧
        frame1 = screen1.capture()
        frame2 = screen2.capture()
        frame3 = screen3.capture()
        
        # 分析用户1和用户2的情绪
        score1, emotions1 = emotion_analyzer.analyze_frame(frame1, 'user1')
        score2, emotions2 = emotion_analyzer.analyze_frame(frame2, 'user2')
        
        # 记录当前时间戳的情绪数据（从开始按钮点击后的相对时间）
        elapsed_time = datetime.now() - st.session_state.start_time
        elapsed_seconds = int(elapsed_time.total_seconds())
        minutes, seconds = divmod(elapsed_seconds, 60)
        current_time = f"{minutes:02d}:{seconds:02d}"
        
        st.session_state.emotion_data['timestamp'].append(current_time)
        st.session_state.emotion_data['user1_score'].append(score1)
        st.session_state.emotion_data['user2_score'].append(score2)
        st.session_state.emotion_data['emotion_diff'].append(abs(score1 - score2))
        
        # 创建情绪光谱图
        spectrum1 = emotion_analyzer.create_emotion_spectrum(emotions1)
        spectrum2 = emotion_analyzer.create_emotion_spectrum(emotions2)
        
        # 更新用户1的视频和情绪光谱
        img_placeholder_1.image(frame1, channels="BGR")
        spectrum_placeholder_1.image(spectrum1, channels="BGR")
        
        # 更新用户2的视频和情绪光谱
        img_placeholder_2.image(frame2, channels="BGR")
        spectrum_placeholder_2.image(spectrum2, channels="BGR")
        
        # 更新用户3的视频
        img_placeholder_3.image(frame3, channels="BGR")
        
        # 更新情绪图表
        times = list(range(len(st.session_state.emotion_data['timestamp'])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user1_score'], 
                                 mode='lines+markers', name='User 1'))
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user2_score'], 
                                 mode='lines+markers', name='User 2'))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=180,  # 减小图表高度
            xaxis_title="Time",
            yaxis_title="Emotion Score",
            yaxis=dict(range=[-1, 1])
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.01)
        
    except Exception as e:
        st.error(f"Error in main loop: {e}")
        break 