import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from deepface import DeepFace
import threading
import time

class EmotionAnalyzer:
    def __init__(self, history_length=30):
        """初始化情绪分析器"""
        # 用于存储每个用户的情绪历史
        self.emotion_history = defaultdict(lambda: deque(maxlen=history_length))
        self.emotion_scores = defaultdict(lambda: deque(maxlen=history_length))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # DeepFace可能会在第一次调用时下载模型，这可能会导致延迟
        # 创建一个后台线程来预热模型
        self.models_ready = False
        threading.Thread(target=self._preload_models).start()
        
        # 情绪映射，将DeepFace的情绪映射到我们需要的格式
        self.emotion_mapping = {
            'happy': 'happy',
            'sad': 'sad',
            'neutral': 'neutral',
            'angry': 'sad',     # 将angry映射到sad类别
            'fear': 'sad',      # 将fear映射到sad类别
            'surprise': 'happy', # 将surprise映射到happy类别
            'disgust': 'sad'     # 将disgust映射到sad类别
        }
        
    def _preload_models(self):
        """预加载DeepFace模型以减少第一次分析的延迟"""
        try:
            # 使用一个空白图像预热模型
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            self.models_ready = True
            print("DeepFace models loaded successfully")
        except Exception as e:
            print(f"Error preloading DeepFace models: {e}")
    
    def analyze_frame(self, frame, user_id):
        """分析视频帧中的情绪
        
        Args:
            frame: 视频帧
            user_id: 用户ID
            
        Returns:
            tuple: (情绪得分, 情绪字典)
        """
        if frame is None:
            return 0, {'neutral': 1.0}
        
        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 如果没有检测到人脸，返回中性情绪
        if len(faces) == 0:
            emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0}
            self.emotion_history[user_id].append(emotions)
            self.emotion_scores[user_id].append(0)
            return 0, emotions
        
        # 选择最大的人脸进行分析
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # 使用DeepFace分析情绪
            if not self.models_ready:
                print("DeepFace models are still loading, using fallback...")
                return self._fallback_emotion(user_id)
                
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            # DeepFace返回的是一个列表，我们取第一个结果
            if isinstance(result, list):
                result = result[0]
                
            # 获取情绪字典
            deepface_emotions = result['emotion']
            
            # 将DeepFace的情绪映射到我们的三种情绪
            emotions = {'happy': 0.0, 'sad': 0.0, 'neutral': 0.0}
            for emotion, score in deepface_emotions.items():
                mapped_emotion = self.emotion_mapping.get(emotion.lower(), 'neutral')
                emotions[mapped_emotion] += score / 100  # DeepFace返回的是百分比
            
            # 归一化
            total = sum(emotions.values())
            for emotion in emotions:
                emotions[emotion] /= total
                
            # 计算情绪得分 (-1 到 1)
            emotion_score = emotions['happy'] - emotions['sad']
            
            # 存储历史
            self.emotion_history[user_id].append(emotions)
            self.emotion_scores[user_id].append(emotion_score)
            
            return emotion_score, emotions
            
        except Exception as e:
            print(f"Error analyzing emotions: {e}")
            return self._fallback_emotion(user_id)
    
    def _fallback_emotion(self, user_id):
        """当DeepFace分析失败时的后备方案"""
        # 如果有历史数据，返回最近的情绪
        if user_id in self.emotion_history and len(self.emotion_history[user_id]) > 0:
            last_emotion = self.emotion_history[user_id][-1]
            last_score = self.emotion_scores[user_id][-1]
            return last_score, last_emotion
        
        # 否则返回中性情绪
        neutral_emotion = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0}
        return 0, neutral_emotion
    
    def draw_emotion_on_frame(self, frame, score, emotions):
        """在视频帧上绘制情绪信息
        
        Args:
            frame: 视频帧
            score: 情绪得分
            emotions: 情绪字典
        
        Returns:
            frame: 添加了情绪信息的视频帧
        """
        if frame is None:
            return None
            
        # 创建一个副本以避免修改原始帧
        result_frame = frame.copy()
        
        # 绘制情绪条
        height, width = result_frame.shape[:2]
        bar_height = 30
        bar_y = height - bar_height - 10
        
        # 绘制背景
        cv2.rectangle(result_frame, (10, bar_y), (width - 10, bar_y + bar_height), (200, 200, 200), -1)
        
        # 计算情绪条的中点和宽度
        mid_x = width // 2
        bar_width = width - 20
        
        # 将得分从[-1,1]映射到[0,bar_width]
        pos_x = int(mid_x + (score * bar_width / 2))
        
        # 绘制左侧（负面情绪）
        cv2.rectangle(result_frame, (10, bar_y), (mid_x, bar_y + bar_height), (0, 0, 255), -1)
        
        # 绘制右侧（正面情绪）
        cv2.rectangle(result_frame, (mid_x, bar_y), (width - 10, bar_y + bar_height), (0, 255, 0), -1)
        
        # 绘制中线
        cv2.line(result_frame, (mid_x, bar_y), (mid_x, bar_y + bar_height), (0, 0, 0), 2)
        
        # 绘制当前情绪指示器
        cv2.circle(result_frame, (pos_x, bar_y + bar_height // 2), 10, (255, 255, 255), -1)
        cv2.circle(result_frame, (pos_x, bar_y + bar_height // 2), 10, (0, 0, 0), 2)
        
        # 添加情绪文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        emotions_text = " ".join([f"{e}: {v:.2f}" for e, v in emotions.items()])
        cv2.putText(result_frame, emotions_text, (10, bar_y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return result_frame
    
    def create_emotion_chart(self, user_id=None):
        """创建情绪变化图表
        
        Args:
            user_id: 用户ID，如果为None则显示所有用户
            
        Returns:
            图表对象
        """
        plt.figure(figsize=(10, 6))
        
        if user_id is not None and user_id in self.emotion_scores:
            # 显示单个用户的情绪变化
            scores = list(self.emotion_scores[user_id])
            plt.plot(scores, label=f'User {user_id}')
        else:
            # 显示所有用户的情绪变化
            for uid, scores in self.emotion_scores.items():
                plt.plot(list(scores), label=f'User {uid}')
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.ylim(-1, 1)
        plt.xlabel('Time')
        plt.ylabel('Emotion Score')
        plt.title('Emotion Changes Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt

    def create_emotion_spectrum(self, emotions):
        """创建情绪光谱图
        
        Args:
            emotions: 情绪字典，包含 'happy', 'sad', 'neutral' 的值
            
        Returns:
            numpy.ndarray: 情绪光谱图像
        """
        # 创建一个长方形图像作为光谱
        height = 30
        width = 300
        spectrum = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 获取情绪值
        happy_val = emotions.get('happy', 0)
        sad_val = emotions.get('sad', 0)
        neutral_val = emotions.get('neutral', 0)
        
        # 归一化确保总和为1
        total = happy_val + sad_val + neutral_val
        if total > 0:
            happy_val /= total
            sad_val /= total
            neutral_val /= total
        
        # 降低亮度系数 (0.0-1.0)
        brightness_factor = 0.6
        
        # 降低饱和度 - 通过向灰色混合
        saturation_factor = 0.7
        gray_component = 1 - saturation_factor
        
        # 创建RGB渐变 - R代表happy, G代表neutral, B代表sad
        for x in range(width):
            # 计算原始颜色值
            r = int(255 * happy_val)
            g = int(255 * neutral_val)
            b = int(255 * sad_val)
            
            # 降低饱和度 (向灰色混合)
            gray = (r + g + b) // 3
            r = int(r * saturation_factor + gray * gray_component)
            g = int(g * saturation_factor + gray * gray_component)
            b = int(b * saturation_factor + gray * gray_component)
            
            # 降低亮度
            r = int(r * brightness_factor)
            g = int(g * brightness_factor)
            b = int(b * brightness_factor)
            
            # 填充整列
            spectrum[:, x] = [r, g, b]
        
        # # 添加文字标签
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # font_color = (200, 200, 200)  # 更柔和的文字颜色
        # font_thickness = 1
        
        # # 添加情绪文本
        # emotions_text = f"happy: {happy_val:.2f} neutral: {neutral_val:.2f} sad: {sad_val:.2f}"
        # text_size = cv2.getTextSize(emotions_text, font, font_scale, font_thickness)[0]
        # text_x = (width - text_size[0]) // 2
        # text_y = (height + text_size[1]) // 2
        
        # cv2.putText(spectrum, emotions_text, (text_x, text_y), 
        #             font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        return spectrum 