# Driver Attention Monitoring Prototype

A C++/OpenCV application that monitors driver attention via webcam. It detects faces, fits facial landmarks, estimates head pose and eye blinks, and issues timed warnings when the driver appears distracted.

## Features

- **Face Detection**: Uses a Caffe ResNet SSD model to find the driver’s face.
- **Facial Landmarks**: Fits 68-point landmarks with OpenCV’s LBF model.
- **Head Pose Estimation**: Computes yaw angle via RANSAC‐based PnP.
- **Blink Detection**: Measures Eye Aspect Ratio (EAR) to detect closed eyes.
- **Smoothing & Hysteresis**: Applies median filtering, EMA smoothing, and hysteresis thresholds for stable state transitions.
- **Timed Warnings**:
  - Warning Level 1 after 5 s of continuous distraction.
  - Warning Level 2 after 20 s total, with a 1 s beep.
  - 5 s lockout after Warning 1 before allowing state changes.
- **Resizable Window**: Opens at 75 % of screen size, centered on primary display.
- **Cross-Platform**: Windows, Linux, and macOS support (Windows‐specific pop‐up code under `#ifdef _WIN32`).

## Prerequisites

- **Operating System**: Windows 10/11, Linux, or macOS
- **Compiler**: Visual Studio 2022 (MSVC) or GCC/Clang with C++17 support
- **Build System**: CMake ≥ 3.20
- **Dependencies**:
  - OpenCV 4.11 with `dnn`, `face`, `highgui`, `videoio`
  - (Optional) **vcpkg** for dependency management

### Installing with vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.bootstrap-vcpkg.bat
.vcpkg install opencv[core,highgui,videoio,dnn,face]:x64-windows
```

## Build Instructions

```powershell
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
cd build\Release
.attention_demo.exe
```

## Usage

1. Place the `models/` folder (with `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`, `lbfmodel.yaml`) next to `build/`.
2. Run the executable (`attention_demo.exe` or `attention_demo`).
3. The window shows:
   - **ATTENTIVE** or **DISTRACTED**.
   - Debug readout of **EAR** and **yaw**.
4. Distract yourself:
   - **5 s** → WARNING LEVEL 1 pop-up.
   - **20 s** → WARNING LEVEL 2 pop-up + beep.
5. After Warning 1, stays in distracted for **5 s** lockout.
6. Press **ESC** or close window to exit.

## Project Structure

```
.
├── include/
│   └── AttentionMonitor.hpp
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── lbfmodel.yaml
├── src/
│   ├── AttentionMonitor.cpp
│   └── main.cpp
├── CMakeLists.txt
└── README.md
```

## Contributing

1. Fork the repo.
2. Create a branch (`git checkout -b feature/foo`).
3. Commit changes.
4. Open a Pull Request.
