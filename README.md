# 情绪分析应用

这个应用程序使用DeepFace分析视频中的人脸情绪，并实时显示情绪变化。

## 安装

1. 克隆或下载此仓库 
2. cd到仓库目录下, 安装依赖项：
   ```
   conda create -n dhh-comm-env python=3.12.4
   conda activate dhh-comm-env
   pip install -r requirements.txt

   ```
   或者使用 (不推荐)
   ```
   conda env create -f environment.yml
   ```
3. 运行应用程序：
   ```
   streamlit run app.py --server.runOnSave=true
   ```

### 虚拟环境管理

常用的 conda 命令：
conda activate dhh-comm-env - 激活环境
conda deactivate - 退出环境
conda env list - 查看所有环境
conda list - 查看当前环境安装的包
conda env export > environment.yml - 导出环境配置
conda remove -n dhh-comm-env --all - 删除环境

## 使用说明

1. 点击"Start/Stop Capture"按钮开始捕获和分析
2. 应用程序将分析屏幕上指定区域的人脸情绪
3. 停止捕获后，数据将保存到logs文件夹中

## 注意事项

- 首次运行时，DeepFace可能需要下载模型文件
- 请确保vid文件夹中包含视频文件 (视频文件需要是mp4格式, 请将视频文件命名为vid.mp4)