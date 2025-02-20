import cv2
import numpy as np
import plotly.graph_objects as go
from collections import deque
from datetime import datetime
import os

class EmotionAnalyzer:
    def __init__(self, history_size=30):
        self.history_size = history_size
        
        # 加载OpenCV的人脸检测器
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            # 如果找不到默认路径，尝试使用相对路径
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                print("Warning: Could not find face cascade file. Face detection will be disabled.")
                self.face_cascade = None
            else:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 为每个用户存储情绪历史
        self.emotion_history = {
            'user1': {
                'timestamps': deque(maxlen=history_size),
                'emotions': deque(maxlen=history_size)
            },
            'user2': {
                'timestamps': deque(maxlen=history_size),
                'emotions': deque(maxlen=history_size)
            }
        }
        
        # 模拟情绪状态
        self.current_emotions = {
            'user1': {'happy': 0.5, 'neutral': 0.5},
            'user2': {'happy': 0.5, 'neutral': 0.5}
        }

    def _simulate_emotion(self, face_detected, user_id):
        """模拟情绪变化"""
        if not face_detected:
            return {'neutral': 1.0}
            
        current = self.current_emotions[user_id]
        
        # 随机调整情绪值
        new_emotions = {}
        for emotion in ['happy', 'neutral', 'sad']:
            if emotion in current:
                # 在当前值的基础上随机波动
                new_value = current[emotion] + np.random.uniform(-0.1, 0.1)
                new_emotions[emotion] = max(0, min(1, new_value))
            else:
                new_emotions[emotion] = np.random.uniform(0, 0.3)
                
        # 归一化
        total = sum(new_emotions.values())
        for emotion in new_emotions:
            new_emotions[emotion] /= total
            
        self.current_emotions[user_id] = new_emotions
        return new_emotions

    def analyze_frame(self, frame, user_id):
        """分析单帧图像中的情绪"""
        try:
            faces = []
            if self.face_cascade is not None:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 检测人脸
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # 模拟情绪分析
            emotions = self._simulate_emotion(len(faces) > 0, user_id)
            
            # 计算情绪得分 (-1 到 1 的范围)
            emotion_score = emotions.get('happy', 0) - emotions.get('sad', 0)
            
            # 更新历史记录
            current_time = datetime.now()
            self.emotion_history[user_id]['timestamps'].append(current_time)
            self.emotion_history[user_id]['emotions'].append(emotion_score)
            
            # 在人脸上画框
            if faces is not None:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            return emotion_score, emotions
            
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return 0, None

    def create_emotion_chart(self):
        """创建情绪变化图表"""
        fig = go.Figure()
        
        # 添加用户1的数据
        times1 = [t.strftime("%H:%M:%S") for t in self.emotion_history['user1']['timestamps']]
        emotions1 = list(self.emotion_history['user1']['emotions'])
        if emotions1:
            fig.add_trace(go.Scatter(
                x=times1,
                y=emotions1,
                mode='lines+markers',
                name='User 1',
                line=dict(color='#1f77b4')
            ))
        
        # 添加用户2的数据
        times2 = [t.strftime("%H:%M:%S") for t in self.emotion_history['user2']['timestamps']]
        emotions2 = list(self.emotion_history['user2']['emotions'])
        if emotions2:
            fig.add_trace(go.Scatter(
                x=times2,
                y=emotions2,
                mode='lines+markers',
                name='User 2',
                line=dict(color='#ff7f0e')
            ))
        
        # 更新布局
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=200,
            xaxis_title="Time",
            yaxis_title="Emotion Score",
            yaxis=dict(range=[-1, 1]),
            showlegend=True
        )
        
        return fig

    def draw_emotion_on_frame(self, frame, emotion_score, emotions_dict):
        """在视频帧上绘制情绪信息"""
        if emotions_dict:
            # 添加情绪分数
            score_text = f"Score: {emotion_score:.2f}"
            cv2.putText(frame, score_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加主要情绪
            y_pos = 60
            for emotion, score in emotions_dict.items():
                text = f"{emotion}: {score:.2f}"
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
        
        return frame 