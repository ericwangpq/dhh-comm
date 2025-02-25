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

# Create layout with tighter spacing
col1, col2 = st.columns([1, 3])

# Left column - participant videos
with col1:
    img_placeholder_1 = st.empty()
    img_placeholder_2 = st.empty()
    img_placeholder_3 = st.empty()

# Right column - main video and diagram
with col2:
    try:
        st.video('vid/vid.mp4')
    except Exception as e:
        st.error(f"Error loading video: {e}")
    
    # Create a sample line chart for emotion tracking
    def create_sample_chart():
        times = [datetime.now().strftime("%H:%M:%S") for _ in range(10)]
        emotions = np.random.rand(10)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=emotions, mode='lines+markers'))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,  # Reduced height for more compact layout
            xaxis_title="Time",
            yaxis_title="Emotion Score"
        )
        return fig
    
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
            return np.zeros((200, 300, 3), dtype=np.uint8)

# Initialize screen captures
# You'll need to adjust these coordinates based on your screen layout
screen1 = ScreenCapture(0, 0, 300, 200)  # User 1 region
screen2 = ScreenCapture(0, 220, 300, 200)  # User 2 region
screen3 = ScreenCapture(0, 440, 300, 200)  # Moderator region

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
        height=200,
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
        
        # Find top 3 moments with highest emotion difference
        df['abs_diff'] = abs(df['user1_score'] - df['user2_score'])
        top_diffs = df.nlargest(3, 'abs_diff')
        
        # Display the peak differences
        st.subheader("Top 3 Emotion Difference Peaks")
        for i, (_, row) in enumerate(top_diffs.iterrows(), 1):
            st.write(f"Peak {i}: At {row['timestamp']}, User 1: {row['user1_score']:.2f}, "
                    f"User 2: {row['user2_score']:.2f}, Difference: {row['abs_diff']:.2f}")
        
        # Reset the data for next session
        st.session_state.emotion_data = {
            'timestamp': [],
            'user1_score': [],
            'user2_score': [],
            'emotion_diff': []
        }

# Main loop to update video feeds
while st.session_state.running:
    try:
        # 直接使用ScreenCapture对象的capture方法获取视频帧
        frame1 = screen1.capture()
        frame2 = screen2.capture()
        frame3 = screen3.capture()
        
        # Analyze emotions for user 1 and 2
        score1, emotions1 = emotion_analyzer.analyze_frame(frame1, 'user1')
        score2, emotions2 = emotion_analyzer.analyze_frame(frame2, 'user2')
        
        # Record emotion data with current timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.emotion_data['timestamp'].append(current_time)
        st.session_state.emotion_data['user1_score'].append(score1)
        st.session_state.emotion_data['user2_score'].append(score2)
        st.session_state.emotion_data['emotion_diff'].append(abs(score1 - score2))
        
        # Draw emotions on frames
        frame1 = emotion_analyzer.draw_emotion_on_frame(frame1, score1, emotions1)
        frame2 = emotion_analyzer.draw_emotion_on_frame(frame2, score2, emotions2)
        
        # Update video displays
        img_placeholder_1.image(frame1, channels="BGR")
        img_placeholder_2.image(frame2, channels="BGR")
        img_placeholder_3.image(frame3, channels="BGR")
        
        # Update emotion chart
        times = list(range(len(st.session_state.emotion_data['timestamp'])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user1_score'], 
                                 mode='lines+markers', name='User 1'))
        fig.add_trace(go.Scatter(x=times, y=st.session_state.emotion_data['user2_score'], 
                                 mode='lines+markers', name='User 2'))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,
            xaxis_title="Time",
            yaxis_title="Emotion Score",
            yaxis=dict(range=[-1, 1])
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.1)
        
    except Exception as e:
        st.error(f"Error in main loop: {e}")
        break 