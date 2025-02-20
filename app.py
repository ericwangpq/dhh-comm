import streamlit as st
import cv2
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pyautogui
import time
from emotion_analyzer import EmotionAnalyzer

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
    chart_placeholder.plotly_chart(create_sample_chart(), use_container_width=True)

# Screen capture settings
class ScreenCapture:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def capture(self):
        # Capture the specified region of the screen
        screenshot = pyautogui.screenshot(region=(self.x, self.y, self.width, self.height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

# Initialize screen captures
# You'll need to adjust these coordinates based on your screen layout
screen1 = ScreenCapture(0, 0, 300, 200)  # User 1 region
screen2 = ScreenCapture(0, 220, 300, 200)  # User 2 region
screen3 = ScreenCapture(0, 440, 300, 200)  # Moderator region

# Replace get_video_feed with screen capture
def get_video_feed(screen_capture):
    try:
        return screen_capture.capture()
    except Exception as e:
        st.error(f"Error capturing screen: {e}")
        return np.zeros((200, 300, 3), dtype=np.uint8)

# Add a start/stop button
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button('Start/Stop Capture'):
    st.session_state.running = not st.session_state.running

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# Main loop to update video feeds
while st.session_state.running:
    try:
        # Update video feeds
        frame1 = get_video_feed(screen1)
        frame2 = get_video_feed(screen2)
        frame3 = get_video_feed(screen3)
        
        # Analyze emotions for user 1 and 2
        score1, emotions1 = emotion_analyzer.analyze_frame(frame1, 'user1')
        score2, emotions2 = emotion_analyzer.analyze_frame(frame2, 'user2')
        
        # Draw emotions on frames
        frame1 = emotion_analyzer.draw_emotion_on_frame(frame1, score1, emotions1)
        frame2 = emotion_analyzer.draw_emotion_on_frame(frame2, score2, emotions2)
        
        # Update video displays
        img_placeholder_1.image(frame1, channels="BGR")
        img_placeholder_2.image(frame2, channels="BGR")
        img_placeholder_3.image(frame3, channels="BGR")
        
        # Update emotion chart
        chart_placeholder.plotly_chart(emotion_analyzer.create_emotion_chart(), 
                                     use_container_width=True)
        
        time.sleep(0.1)
        
    except Exception as e:
        st.error(f"Error in main loop: {e}")
        break 