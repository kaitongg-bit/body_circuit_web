#!/usr/bin/env python3
"""
测试摄像头连接和可用性
"""
import cv2

def test_camera():
    print("正在测试摄像头...")
    print("-" * 50)
    
    # 尝试多个摄像头索引
    for camera_index in range(5):
        print(f"\n尝试摄像头索引 {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ 摄像头 {camera_index} 可用!")
                print(f"   分辨率: {width}x{height}")
                
                # 显示测试画面
                cv2.imshow(f'Camera {camera_index} Test', frame)
                print("   按任意键继续测试下一个摄像头...")
                cv2.waitKey(2000)  # 显示2秒
                cv2.destroyAllWindows()
            else:
                print(f"❌ 摄像头 {camera_index} 打开了但无法读取帧")
            cap.release()
        else:
            print(f"❌ 摄像头 {camera_index} 无法打开")
    
    print("\n" + "-" * 50)
    print("摄像头测试完成")
    print("\n提示:")
    print("1. 如果所有摄像头都失败，请检查系统设置 -> 隐私与安全性 -> 摄像头")
    print("2. 确保授予 Terminal 或 Python 访问摄像头的权限")
    print("3. 关闭其他可能使用摄像头的应用（如 Zoom、FaceTime 等）")

if __name__ == "__main__":
    test_camera()
