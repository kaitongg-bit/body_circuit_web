#!/bin/bash

echo "=================================="
echo "身体回路乐队 - 快速启动脚本"
echo "=================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

echo "✅ Python 版本:"
python3 --version
echo ""

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "✅ 虚拟环境已存在"
fi
echo ""

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖包..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ 依赖安装完成"
echo ""

# 生成音频样本（如果不存在）
if [ ! -d "audio_samples" ] || [ ! -f "audio_samples/drum.wav" ]; then
    echo "🎵 生成音频样本..."
    python generate_audio_samples.py
    echo ""
else
    echo "✅ 音频样本已存在"
    echo ""
fi

# 运行程序
echo "🚀 启动身体回路乐队..."
echo "=================================="
echo ""
python body_circuit_band_full.py

# 退出时清理
deactivate
