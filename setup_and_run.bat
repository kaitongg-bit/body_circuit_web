@echo off
echo ==================================
echo 身体回路乐队 - 快速启动脚本
echo ==================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python
    echo 请先安装 Python 3.8 或更高版本
    pause
    exit /b 1
)

echo ✅ Python 已安装
python --version
echo.

REM 创建虚拟环境
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
    echo ✅ 虚拟环境创建完成
) else (
    echo ✅ 虚拟环境已存在
)
echo.

REM 激活虚拟环境
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo 📥 安装依赖包...
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo ✅ 依赖安装完成
echo.

REM 生成音频样本
if not exist "audio_samples\drum.wav" (
    echo 🎵 生成音频样本...
    python generate_audio_samples.py
    echo.
) else (
    echo ✅ 音频样本已存在
    echo.
)

REM 运行程序
echo 🚀 启动身体回路乐队...
echo ==================================
echo.
python body_circuit_band_full.py

REM 退出时清理
call deactivate
pause
