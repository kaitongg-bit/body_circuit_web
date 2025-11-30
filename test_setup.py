"""
环境测试脚本
用于验证所有依赖是否正确安装
"""

import sys

def test_imports():
    """测试所有必需的库导入"""
    print("🔍 测试 Python 版本...")
    print(f"   Python {sys.version}")
    if sys.version_info < (3, 8):
        print("   ❌ 需要 Python 3.8 或更高版本")
        return False
    else:
        print("   ✅ Python 版本符合要求")

    tests = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pygame", "pygame"),
        ("SciPy", "scipy"),
        ("MediaPipe", "mediapipe"),
    ]

    optional_tests = [
        ("Ultralytics (可选)", "ultralytics"),
    ]

    print("\n🔍 测试必需库...")
    all_passed = True

    for name, module in tests:
        try:
            __import__(module)
            version = ""
            try:
                mod = sys.modules[module]
                if hasattr(mod, '__version__'):
                    version = f" (v{mod.__version__})"
            except:
                pass
            print(f"   ✅ {name}{version}")
        except ImportError:
            print(f"   ❌ {name} - 未安装")
            all_passed = False

    print("\n🔍 测试可选库...")
    for name, module in optional_tests:
        try:
            __import__(module)
            mod = sys.modules[module]
            version = ""
            if hasattr(mod, '__version__'):
                version = f" (v{mod.__version__})"
            print(f"   ✅ {name}{version}")
        except ImportError:
            print(f"   ⚠️  {name} - 未安装（完整版功能将不可用）")

    return all_passed


def test_camera():
    """测试摄像头"""
    print("\n🔍 测试摄像头...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"   ✅ 摄像头工作正常 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                print("   ❌ 摄像头无法读取画面")
                return False
        else:
            print("   ❌ 无法打开摄像头")
            return False
    except Exception as e:
        print(f"   ❌ 摄像头测试失败: {e}")
        return False


def test_audio():
    """测试音频"""
    print("\n🔍 测试音频系统...")
    try:
        import pygame
        pygame.mixer.init()
        print("   ✅ 音频系统初始化成功")
        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f"   ❌ 音频系统测试失败: {e}")
        return False


def check_audio_files():
    """检查音频文件"""
    print("\n🔍 检查音频文件...")
    import os

    files = [
        "audio_samples/drum.wav",
        "audio_samples/bass.wav",
        "audio_samples/harmony.wav"
    ]

    all_exist = True
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} - 不存在")
            all_exist = False

    if not all_exist:
        print("\n   💡 提示: 运行 'python generate_audio_samples.py' 生成音频文件")

    return all_exist


def test_pose_detection():
    """测试姿态检测"""
    print("\n🔍 测试姿态检测模型...")

    # 测试 MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        print("   ✅ MediaPipe Pose 初始化成功")
        pose.close()
    except Exception as e:
        print(f"   ❌ MediaPipe Pose 初始化失败: {e}")
        return False

    # 测试 YOLO (可选)
    try:
        from ultralytics import YOLO
        print("   ℹ️  正在下载 YOLOv8-Pose 模型（首次运行）...")
        model = YOLO('yolov8n-pose.pt')
        print("   ✅ YOLOv8-Pose 加载成功")
    except ImportError:
        print("   ⚠️  Ultralytics 未安装 - 完整版将不可用")
    except Exception as e:
        print(f"   ⚠️  YOLOv8-Pose 加载失败: {e}")

    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("  身体回路乐队 - 环境测试")
    print("=" * 60)

    results = {
        "依赖库": test_imports(),
        "摄像头": test_camera(),
        "音频系统": test_audio(),
        "音频文件": check_audio_files(),
        "姿态检测": test_pose_detection(),
    }

    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed and name in ["依赖库", "摄像头"]:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 所有核心功能测试通过！")
        print("\n📝 下一步:")
        print("   1. 如果音频文件不存在，运行: python generate_audio_samples.py")
        print("   2. 运行简化版测试: python body_circuit_band_simple.py")
        print("   3. 运行完整版: python body_circuit_band_full.py")
    else:
        print("\n❌ 部分测试未通过，请检查上述问题")
        print("\n💡 建议:")
        print("   1. 运行: pip install -r requirements.txt")
        print("   2. 检查摄像头权限")
        print("   3. 重新运行此测试脚本")

    print()


if __name__ == "__main__":
    main()
