# Cover Studio · Windows 本地一键部署版 (RTX 3060 6GB 优化)

[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11%20x64-0078D6?logo=windows)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.12%20(64--bit)-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu126-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

将原本依赖 Google Colab 的 **[tsumugi-piano-cover](https://github.com/anime-song/tsumugi-piano-cover)** 与 **[tsumugi](https://github.com/anime-song/tsumugi)** 钢琴翻弹生成系统，重构为 **Windows 10/11 原生一键安装与启动方案**，并针对 **6GB 显存显卡（如 RTX 3060 Laptop 6GB）** 进行了全链路兼容性与显存优化。

---

## ✨ 核心特性与修复亮点

1. **真正的一键部署 (`01_install.bat`)**
   - 自动拉取两个上游开源仓库（`tsumugi-piano-cover` 与 `tsumugi`），并通过 `uv` 自动构建两套独立的虚拟环境。
   - **CUDA 12.6 驱动兼容补丁**：自动将上游写死的 `cu130` 依赖切换为官方 `torch==2.13.0+cu126` 索引并重新解析 `uv.lock`，无需升级测试版显卡驱动或单独安装庞大的 CUDA Toolkit。
2. **内置 FFmpeg 7.1.1 Shared 与 TorchCodec 解码修复**
   - 自动下载并校验（SHA256）官方 `FFmpeg 7.1.1 full_build-shared` 动态链接库到项目内部目录，隔离系统环境变量中残缺 DLL（如 `scrcpy`）的干扰。
   - 自动修复上游 `tsumugi_worker.py` 中 `os.add_dll_directory()` 句柄提前释放导致 `libtorchcodec_core*.dll` 加载失败的 Bug。
3. **Hugging Face 模型稳定下载（防 `0/3` 卡死）**
   - 默认启用 `HF_HUB_DISABLE_XET=1` 兼容模式与超时重试，解决 Windows 网络环境下拉取 `SheetSage2-Refined`（~2.7 GB）与 `V2` 模型时卡在 `Fetching 3 files: 0/3` 的问题。
4. **6GB 显存多档位调节与 CPU 转录兜底 (`03_toolbox.bat`)**
   - 将上游默认容易爆显存的 `window-batch-size=4` 调整为可持久化的 4 档配置（`1=稳定` / `2=均衡` / `3=高速` / `4=激进`），并提供一键切换至 **CPU 转录 + GPU 翻弹** 混合模式。

---

## 🛠️ 环境准备（仅需一次）

| 依赖项 | 要求说明 | 检查命令 / 下载地址 |
| :--- | :--- | :--- |
| **操作系统** | Windows 10 / 11 (64 位) | - |
| **NVIDIA 显卡** | 显存 ≥ 6GB，驱动版本建议 **≥ 560.76**（`nvidia-smi` 显示 CUDA ≥ 12.6 即可） | `nvidia-smi` / [NVIDIA 官方驱动下载](https://www.nvidia.com/Download/index.aspx) |
| **Python 3.12** | 必须安装 **64 位 Python 3.12**，安装时勾选 **Python Launcher (`py`)** | `py -3.12 -V` / [Python 3.12 下载](https://www.python.org/downloads/release/python-31210/) |
| **Git** | Git for Windows | `git --version` / [Git 官方下载](https://git-scm.com/downloads/win) |
| **磁盘空间** | 建议所在分区预留 **≥ 30 GiB** 可用空间（用于两套 GPU 环境与模型权重） | - |

---

## 🚀 快速开始（三步跑通）

### 第一步：克隆本仓库
```bash
git clone https://github.com/likalikali/CoverStudio_Win10_RTX3060_6GB.git
cd CoverStudio_Win10_RTX3060_6GB
```

### 第二步：双击 `01_install.bat` 一键部署
脚本将全自动完成以下 6 个步骤：
1. 克隆 `tsumugi-piano-cover` 与 `tsumugi` 源码至项目目录
2. 应用 PyTorch CUDA 12.6 (`cu126`) 源替换补丁
3. 创建并安装 Cover Studio 虚拟环境
4. 创建并同步 `tsumugi` 音轨分离与转录虚拟环境
5. 自动下载校验 `FFmpeg 7.1.1 Shared` DLL 并修补 Worker 句柄
6. 运行 GPU 张量运算自检与 `TorchCodec` / `torchaudio` WAV 解码自检

### 第三步：双击 `02_start.bat` 启动服务
- 启动后在浏览器访问：**http://127.0.0.1:8100**
- 首次生成或转录时会自动下载模型权重至 `cache/huggingface/`（也可先通过 `03_toolbox.bat` 预下载）。
- 保持命令行窗口开启；按 `Ctrl+C` 或关闭窗口即可停止服务。

---

## 🧰 实用工具箱 (`03_toolbox.bat`)

双击 `03_toolbox.bat` 可打开交互式维护与排错菜单：

```text
[1] 环境全面自检 (检测 GPU / CUDA 12.6 / FFmpeg / TorchCodec)
[2] 切换显存性能档位 (设置转录 window-batch-size: 1~4)
[3] 预下载/修复模型权重 (SheetSage2 & V2，禁用 Xet 防卡死)
[4] 下载/修复 FFmpeg Shared 7.1.1 (解决 TorchCodec 解码报错)
[5] 启动服务 - CPU 转录兼容模式 (解决 6GB 显存转录 OOM)
```

也可以使用命令行直接调用 `windows_cover_studio.py`：
```bash
py -3.12 windows_cover_studio.py install                 # 一键安装
py -3.12 windows_cover_studio.py check                   # 环境与解码自检
py -3.12 windows_cover_studio.py start --mode gpu        # GPU 模式启动 (默认)
py -3.12 windows_cover_studio.py start --mode transcribe-cpu # CPU 转录 + GPU 翻弹
py -3.12 windows_cover_studio.py performance --set 2     # 切换转录 batch 档位 (1~4)
py -3.12 windows_cover_studio.py download-models         # 预下载全部核心模型权重
py -3.12 windows_cover_studio.py fix-ffmpeg              # 单独修复 FFmpeg 与 TorchCodec
```

---

## 📂 项目目录结构

```text
.
├── 01_install.bat                          # 一键部署安装入口
├── 02_start.bat                            # 一键启动服务入口 (GPU 推荐模式)
├── 03_toolbox.bat                          # 交互式工具箱 (自检/档位/模型预下载/修复)
├── windows_cover_studio.py                 # 核心安装、补丁、自检与启动控制器
├── cover_studio_win10_local_3060_6GB.ipynb # 可选 Jupyter Notebook 交互启动入口
├── README.md                               # 项目说明文档
├── LICENSE                                 # MIT 开源协议
├── .gitignore                              # Git 忽略规则 (排除运行环境、缓存与个人数据)
│
├── tsumugi-piano-cover/                    # [运行时自动生成] 上游源码、虚拟环境与内置 FFmpeg
├── cache/huggingface/                      # [运行时自动生成] Hugging Face 模型权重缓存
├── data/projects/                          # [运行时自动生成] 用户上传的音频与生成的 MIDI 项目
└── logs/                                   # [运行时自动生成] 服务端运行日志
```

---

## 💡 6GB 显存（RTX 3060 Laptop）使用建议

- **性能档位选择**：默认 `batch=1` 最稳妥；日常在 RTX 3060 6GB 上推荐通过 `03_toolbox.bat -> [2]` 切换为 **`batch=2`（均衡模式）**。
- **首次测试建议**：建议先将生成时长设为 `30s`、生成数量设为 `1 take` 测试跑通，再生成完整曲目。
- **遇到转录显存不足 (CUDA OOM)**：关闭后台占用显存的浏览器硬件加速或游戏，或在 `03_toolbox.bat` 中选择 `[2]` 调回 `batch=1`，或选择 `[5]` 使用 **CPU 转录兼容模式**（仅转录走 CPU，翻弹生成依然使用 RTX 3060 GPU）。

---

## 📜 致谢与开源许可

- 本仓库仅包含 Windows 平台自动化部署与启动脚本，采用 [MIT License](LICENSE) 开源。
- 核心算法与 Web 界面来自上游开源项目：
  - [anime-song/tsumugi-piano-cover](https://github.com/anime-song/tsumugi-piano-cover)
  - [anime-song/tsumugi](https://github.com/anime-song/tsumugi)
- 预训练模型权重（`SheetSage2-Refined`、`tsumugi-piano-cover-v2` 等）遵循上游 **CC BY-NC 4.0** 协议（仅供非商业用途）。
