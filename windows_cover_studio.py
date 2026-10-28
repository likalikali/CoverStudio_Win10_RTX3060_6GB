# -*- coding: utf-8 -*-
"""Cover Studio - Windows 10/11 (RTX 3060 6GB / CUDA 12.6) One-Click Installer & Launcher.

Commands:
  py -3.12 windows_cover_studio.py install
  py -3.12 windows_cover_studio.py start [--mode gpu|transcribe-cpu]
  py -3.12 windows_cover_studio.py check
  py -3.12 windows_cover_studio.py fix-ffmpeg
  py -3.12 windows_cover_studio.py performance [--set 1|2|3|4]
  py -3.12 windows_cover_studio.py download-models
  py -3.12 windows_cover_studio.py toolbox

This is a lightweight installer and launcher. Model inference is performed by the upstream
anime-song/tsumugi-piano-cover and anime-song/tsumugi projects.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import wave
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
COVER = BASE / "tsumugi-piano-cover"
TSUMUGI = COVER / ".tsumugi" / "tsumugi"
COVER_PY = COVER / ".venv" / "Scripts" / "python.exe"
TSUMUGI_PY = TSUMUGI / ".venv" / "Scripts" / "python.exe"
LOCAL_FFMPEG_BIN = COVER / ".tsumugi" / "ffmpeg-shared" / "bin"
WORKER = COVER / "cover_studio" / "tsumugi_worker.py"
BATCH_CONFIG = BASE / ".cover_batch_size"
HF_CACHE = BASE / "cache" / "huggingface"
DATA = BASE / "data" / "projects"
LOGS = BASE / "logs"
PORT = 8100

# Official FFmpeg 7.1.1 full_build-shared archive & SHA256 from Winget manifest (Gyan.FFmpeg.Shared)
FFMPEG_URL = "https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.zip"
FFMPEG_SHA256 = "9f28727e8b472a04c1d2e520aaa425dca82721b995139b35710091130ea6e699"

BATCH_OPTIONS = {
    1: "稳定模式 (batch=1，6GB 显存最安全，默认)",
    2: "均衡模式 (batch=2，RTX 3060 6GB 推荐日常使用)",
    3: "高速模式 (batch=3，仅在 batch=2 稳定不爆显存时尝试)",
    4: "激进模式 (batch=4，上游默认值，6GB 极易 OOM 爆显存)",
}

MODEL_REPOS = [
    "anime-song/SheetSage2-Refined",
    "anime-song/tsumugi-piano-cover-v2",
]


def say(message: str) -> None:
    print(message, flush=True)


def check_system() -> None:
    if os.name != "nt":
        raise RuntimeError("此启动包仅支持原生 Windows 10/11，不能直接在 Colab、Linux 或 WSL 中运行。")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("请安装 Python 3.12 (64 位)，并使用 'py -3.12 windows_cover_studio.py ...' 运行。")
    if sys.maxsize <= 2**32:
        raise RuntimeError("需要 64 位 Python 3.12。")


def call(cmd: list[str | os.PathLike[str]], *, cwd: Path | None = None) -> None:
    pretty = subprocess.list2cmdline([str(x) for x in cmd])
    say(f"\n▶ {pretty}")
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd, check=False)
    if proc.returncode:
        raise RuntimeError(f"命令失败 (退出码 {proc.returncode}): {pretty}")


def capture(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def nvidia_report() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        say("⚠ 未找到 nvidia-smi：请检查 NVIDIA 显卡驱动和系统 PATH。")
        return False
    result = capture([
        exe,
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode:
        say("⚠ nvidia-smi 无法读取显卡信息：" + result.stdout[-800:])
        return False
    say("检测到 NVIDIA GPU (名称, 驱动版本, 显存总量MiB, 剩余MiB)：\n  " + result.stdout.strip())
    line = result.stdout.strip().splitlines()[0]
    fields = [x.strip() for x in line.split(",")]
    if len(fields) > 1:
        try:
            version = tuple(int(part) for part in fields[1].split(".")[:2])
            if version < (560, 76):
                say("⚠ 当前显卡驱动较旧：CUDA 12.6 版 PyTorch 建议 Windows 驱动版本 >= 560.76。")
            else:
                say("✓ 显卡驱动满足 CUDA 12.6 版 PyTorch 要求（无需单独安装 CUDA Toolkit）。")
        except ValueError:
            pass
    return True


def find_uv() -> Path | None:
    candidates = [
        shutil.which("uv"),
        str(Path(sysconfig.get_path("scripts")) / "uv.exe"),
        str(Path(sys.executable).parent / "Scripts" / "uv.exe"),
    ]
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value)
    return None


def get_uv() -> Path:
    uv = find_uv()
    if uv:
        return uv
    say("正在安装 uv 包管理器……")
    call([sys.executable, "-m", "pip", "install", "--upgrade", "uv"])
    uv = find_uv()
    if uv is None:
        raise RuntimeError("已尝试安装 uv，但未找到 uv.exe。请执行 'py -3.12 -m pip install uv' 后重试。")
    return uv


def ensure_clone(url: str, folder: Path) -> None:
    if (folder / ".git").is_dir():
        say(f"✓ 已存在仓库：{folder}（保留本地补丁，跳过 git clone）")
        return
    if folder.exists():
        raise RuntimeError(f"目标路径已存在但不是 Git 仓库：{folder}。请检查或清理该目录后重试。")
    folder.parent.mkdir(parents=True, exist_ok=True)
    call(["git", "clone", "--depth", "1", url, str(folder)])


def patch_cuda_126_sources() -> None:
    """Switch both upstream projects from cu130 to official PyTorch CUDA 12.6 wheels."""
    requirements = COVER / "requirements.txt"
    project = TSUMUGI / "pyproject.toml"
    if not requirements.is_file() or not project.is_file():
        raise RuntimeError("缺少上游依赖配置文件，无法应用 CUDA 12.6 兼容补丁。")

    source = requirements.read_text(encoding="utf-8")
    if "torch==2.13.0+cu130" in source and "https://download.pytorch.org/whl/cu130" in source:
        fixed = source.replace("torch==2.13.0+cu130", "torch==2.13.0+cu126")
        fixed = fixed.replace("https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu126")
        requirements.write_text(fixed, encoding="utf-8")
        say("✓ Cover Studio requirements.txt 已切换为 torch 2.13.0+cu126")
    elif "torch==2.13.0+cu126" in source and "https://download.pytorch.org/whl/cu126" in source:
        say("✓ Cover Studio 已配置为 CUDA 12.6")
    else:
        raise RuntimeError("Cover Studio 上游 requirements.txt 格式变化，请手动检查 CUDA 索引配置。")

    source = project.read_text(encoding="utf-8")
    if "pytorch-cu130" in source and "https://download.pytorch.org/whl/cu130" in source:
        fixed = source.replace("pytorch-cu130", "pytorch-cu126")
        fixed = fixed.replace("https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu126")
        project.write_text(fixed, encoding="utf-8")
        say("✓ tsumugi pyproject.toml 已切换为 cu126 索引")
    elif "pytorch-cu126" in source and "https://download.pytorch.org/whl/cu126" in source:
        say("✓ tsumugi 已配置为 CUDA 12.6")
    else:
        raise RuntimeError("tsumugi 上游 pyproject.toml 格式变化，请手动检查 CUDA 索引配置。")


def get_window_batch_size() -> int:
    """Read local performance profile; default safely to batch=1."""
    if not BATCH_CONFIG.exists():
        return 1
    try:
        value = int(BATCH_CONFIG.read_text(encoding="ascii").strip())
        if value not in BATCH_OPTIONS:
            raise ValueError(f"unsupported batch size: {value}")
        return value
    except (ValueError, OSError) as exc:
        say(f"⚠ 性能档位配置无效 ({exc})，已回退至安全档位 batch=1。")
        return 1


def patch_worker_batch_size() -> bool:
    """Apply persistent window-batch-size to tsumugi_worker.py."""
    if not WORKER.is_file():
        say("⚠ 未找到 tsumugi_worker.py，跳过 batch-size 调整。")
        return False
    selected = get_window_batch_size()
    source = WORKER.read_text(encoding="utf-8")
    pattern = r"(parser\.add_argument\(\s*['\"]--window-batch-size['\"]\s*,\s*type\s*=\s*int\s*,\s*default\s*=\s*)(\d+)(\s*\))"
    match = re.search(pattern, source)
    if not match:
        say("⚠ tsumugi_worker.py 参数结构变化，未找到 --window-batch-size。")
        return False
    actual = int(match.group(2))
    if actual == selected:
        say(f"✓ 转录性能档位已生效：window-batch-size={selected}")
        return True
    changed = re.sub(pattern, lambda m: m.group(1) + str(selected) + m.group(3), source, count=1)
    WORKER.write_text(changed, encoding="utf-8")
    say(f"✓ 已更新转录性能档位：window-batch-size={actual} -> {selected}")
    return True


def patch_worker_ffmpeg_handle() -> None:
    """Keep Windows DLL directory handle alive during transcription so TorchCodec can load FFmpeg DLLs."""
    if not WORKER.is_file():
        return
    source = WORKER.read_text(encoding="utf-8")
    old = "        os.add_dll_directory(str(Path(args.ffmpeg_dir).resolve()))"
    new = "        _ffmpeg_dll_handle = os.add_dll_directory(str(Path(args.ffmpeg_dir).resolve()))"
    if new in source:
        say("✓ Worker FFmpeg DLL 句柄生命周期补丁已就绪")
        return
    if old in source:
        WORKER.write_text(source.replace(old, new, 1), encoding="utf-8")
        say("✓ 已应用 Windows FFmpeg DLL 加载句柄修复补丁")


def patch_midi_export_fix() -> None:
    """Prevent negative frame onsets from underflowing symusic uint32 MIDI delta-times (0 notes / exploded timeline)."""
    beat_tok = COVER / "piano_ar" / "beat_tokenizer.py"
    if beat_tok.is_file():
        text = beat_tok.read_text(encoding="utf-8")
        old_bt = "        events[:, ONSET] = np.round(onset)"
        new_bt = "        events[:, ONSET] = np.maximum(0, np.round(onset))"
        if old_bt in text and new_bt not in text:
            beat_tok.write_text(text.replace(old_bt, new_bt, 1), encoding="utf-8")
            say("✓ 已应用 BeatTokenizer 负时间戳截断补丁 (防 MIDI 导出损坏)")

    tok_file = COVER / "piano_ar" / "tokenizer.py"
    if tok_file.is_file():
        text = tok_file.read_text(encoding="utf-8")
        old_tok = (
            "        for onset, kind, pitch, duration, velocity in events.tolist():\n"
            "            if kind == KIND_NOTE:\n"
            '                track.notes.append(Note(onset / fr, duration / fr, pitch, velocity, "second"))'
        )
        new_tok = (
            "        for onset, kind, pitch, duration, velocity in events.tolist():\n"
            "            onset = max(0, onset)\n"
            "            if kind == KIND_NOTE:\n"
            '                track.notes.append(Note(onset / fr, max(1, duration) / fr, pitch, velocity, "second"))'
        )
        if old_tok in text and "onset = max(0, onset)" not in text:
            tok_file.write_text(text.replace(old_tok, new_tok, 1), encoding="utf-8")
            say("✓ 已应用 PianoTokenizer MIDI 导出非负时间保护补丁")


def patch_web_midi_player() -> None:
    """Fix frontend MIDI silence: instant multi-harmonic piano synth while loading + dual-CDN fast sample loading."""
    static_dir = COVER / "cover_studio" / "static" / "assets"
    if not static_dir.is_dir():
        try:
            from urllib.request import urlopen
            # Ensure web bundle is downloaded if missing
            if COVER_PY.is_file():
                capture([str(COVER_PY), "-m", "cover_studio.web"], cwd=COVER)
        except Exception:
            pass
    if not static_dir.is_dir():
        return

    bt = "`"
    old_mr = (
        f"function Mr(e,t,n){{return Gn(this,null,function*(){{t=t.replace(/#/g,{bt}%23{bt}).replace(/ /g,{bt}%20{bt}).replace(/([^:]\\/)\\/+/g,{bt}$1{bt});"
        f"let r=yield n.fetch(t);if(r.status!==200){{console.warn({bt}Error loading buffer. Invalid status: {bt},r.status,t);return}}"
        f"try{{let t=yield r.arrayBuffer();return yield e.decodeAudioData(t)}}catch(e){{console.warn({bt}Error loading buffer{bt},e,t)}}}})}}"
    )
    new_mr = (
        f"function Mr(e,t,n){{return Gn(this,null,function*(){{t=t.replace(/#/g,{bt}%23{bt}).replace(/ /g,{bt}%20{bt}).replace(/([^:]\\/)\\/+/g,{bt}$1{bt});"
        f"let u=t.replace({bt}https://smpldsnds.github.io/sfzinstruments-splendid-grand-piano/samples{bt},{bt}https://cdn.jsdelivr.net/gh/smpldsnds/sfzinstruments-splendid-grand-piano@master/samples{bt}),r;"
        f"for(let s of[u,t]){{let c=new AbortController,i=setTimeout(()=>c.abort(),8e3);try{{let o=yield fetch(s,{{signal:c.signal}});if(o&&o.status===200){{r=o;break}}}}catch{{}}finally{{clearTimeout(i)}}}}"
        f"if(!r||r.status!==200){{console.warn({bt}Error loading buffer: {bt},t);return}}"
        f"try{{let t=yield r.arrayBuffer();return yield e.decodeAudioData(t)}}catch(e){{console.warn({bt}Error loading buffer{bt},e,t)}}}})}}"
    )

    old_load = (
        f"loadPiano(){{try{{let e=ma(this.ctx,{{destination:this.coverGain,storage:new er({bt}cover-studio-piano{bt}),"
        f"onLoadProgress:e=>{{this.pianoProgress=e.total?e.loaded/e.total:0,this.emit()}}}});"
    )
    new_load = (
        f"loadPiano(){{try{{let e=ma(this.ctx,{{destination:this.coverGain,storage:new er({bt}cover-studio-piano-v2{bt}),"
        f"notesToLoad:{{notes:[23,29,35,40,45,50,55,60,65,71,76,81,86,91,96,102,108],velocityRange:[41,100]}},"
        f"onLoadProgress:e=>{{this.pianoProgress=e.total?e.loaded/e.total:0,this.emit()}}}});"
    )

    old_play = (
        f"playNote(e,t){{let n=Math.max(.05,e[4]-e[0]);if(this.piano&&this.pianoState==={bt}ready{bt}){{this.piano.start({{note:e[2],velocity:e[3],time:t,duration:n}});return}}"
        f"if(this.pianoState==={bt}loading{bt})return;let r=this.ctx.createOscillator(),i=this.ctx.createGain();r.type={bt}triangle{bt},r.frequency.value=440*2**((e[2]-69)/12);"
        f"let a=.12*(e[3]/127)**1.5;i.gain.setValueAtTime(0,t),i.gain.linearRampToValueAtTime(a,t+.005),i.gain.setTargetAtTime(0,t+.005,.6),i.gain.setTargetAtTime(0,t+n,.08),"
        f"r.connect(i).connect(this.coverGain),r.start(t),r.stop(t+n+.5),this.fallbackVoices.add(r),r.onended=()=>this.fallbackVoices.delete(r)}}"
    )
    new_play = (
        f"playNote(e,t){{let n=Math.max(.05,e[4]-e[0]);if(this.piano&&this.pianoState==={bt}ready{bt}&&this.piano.loadProgress.loaded>0){{this.piano.start({{note:e[2],velocity:e[3],time:t,duration:n}});return}}"
        f"let f=440*2**((e[2]-69)/12),a=.36*(Math.max(1,e[3])/127)**1.15,i=this.ctx.createGain();i.gain.setValueAtTime(1e-4,t),i.gain.linearRampToValueAtTime(a,t+.004),"
        f"i.gain.setTargetAtTime(a*.25,t+.004,.22),i.gain.setTargetAtTime(1e-4,t+n,.07),i.connect(this.coverGain);"
        f"for(let[m,g,tp]of[[1,1,{bt}triangle{bt}],[2,.35,{bt}sine{bt}],[3,.12,{bt}sine{bt}]]){{if(f*m>=this.ctx.sampleRate*.45)continue;"
        f"let r=this.ctx.createOscillator(),gn=this.ctx.createGain();r.type=tp,r.frequency.value=f*m,gn.gain.value=g,r.connect(gn).connect(i),"
        f"r.start(t),r.stop(t+n+.45),this.fallbackVoices.add(r),r.onended=()=>this.fallbackVoices.delete(r)}}}}"
    )

    old_studio = f"(0,p.useEffect)(()=>{{de.data?a.setNotes(de.data.notes):E&&E.state!=={bt}done{bt}&&a.setNotes([])}},[de.data,E?.state,a])"
    new_studio = (
        f"(0,p.useEffect)(()=>{{de.data?a.setNotes(de.data.notes):!E&&c.data?.notes?a.setNotes(c.data.notes.map(e=>[e[0],e[1],e[2],e[3],e[0]+e[1]])):"
        f"E&&E.state!=={bt}done{bt}&&a.setNotes([])}},[de.data,E,c.data,a])"
    )

    for js_file in static_dir.glob("index-*.js"):
        text = js_file.read_text(encoding="utf-8")
        updated = text
        if old_mr in updated:
            updated = updated.replace(old_mr, new_mr, 1)
        if old_load in updated:
            updated = updated.replace(old_load, new_load, 1)
        if old_play in updated:
            updated = updated.replace(old_play, new_play, 1)
        if old_studio in updated:
            updated = updated.replace(old_studio, new_studio, 1)
        if updated != text:
            js_file.write_text(updated, encoding="utf-8")
            say(f"✓ 已应用前端 MIDI 钢琴音源加速与泛音合成器补丁 ({js_file.name})")
        elif "cover-studio-piano-v2" in text:
            say("✓ 前端 MIDI 钢琴音源补丁已就绪")

    app_py = COVER / "cover_studio" / "app.py"
    if app_py.is_file():
        app_text = app_py.read_text(encoding="utf-8")
        old_resp = "        if path and file.is_file() and file.is_relative_to(STATIC_DIR.resolve()):\n            return FileResponse(file)"
        new_resp = '        if path and file.is_file() and file.is_relative_to(STATIC_DIR.resolve()):\n            return FileResponse(file, headers={"Cache-Control": "no-cache"})'
        if old_resp in app_text:
            app_py.write_text(app_text.replace(old_resp, new_resp, 1), encoding="utf-8")


def is_shared_ffmpeg(path: Path) -> bool:
    """Require a complete FFmpeg shared installation (avoiding partial DLLs like scrcpy)."""
    try:
        return (
            path.is_dir()
            and (path / "ffmpeg.exe").is_file()
            and (path / "ffprobe.exe").is_file()
            and all(
                any(path.glob(pattern))
                for pattern in (
                    "avcodec-*.dll",
                    "avformat-*.dll",
                    "avutil-*.dll",
                    "swresample-*.dll",
                    "swscale-*.dll",
                )
            )
        )
    except OSError:
        return False


def ffmpeg_711_is_complete(folder: Path) -> bool:
    return (folder / "ffmpeg.exe").is_file() and all(
        (folder / name).is_file()
        for name in (
            "avcodec-61.dll",
            "avformat-61.dll",
            "avutil-59.dll",
            "swresample-5.dll",
            "swscale-8.dll",
        )
    )


def ffmpeg_shared_bin() -> Path | None:
    candidates: list[Path] = [LOCAL_FFMPEG_BIN]
    override = os.environ.get("FFMPEG_SHARED_BIN")
    if override:
        candidates.append(Path(override))
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            candidates.append(Path(part))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        pkgroot = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if pkgroot.is_dir():
            for pkg in pkgroot.glob("Gyan.FFmpeg.Shared_*"):
                for folder in pkg.glob("*/bin"):
                    candidates.append(folder)
                for folder in pkg.glob("*/*/bin"):
                    candidates.append(folder)
                for folder in pkg.glob("bin"):
                    candidates.append(folder)
    programdata = os.environ.get("PROGRAMDATA")
    if programdata:
        for folder in (Path(programdata) / "chocolatey" / "lib" / "ffmpeg-shared" / "tools").glob("*/bin"):
            candidates.append(folder)
    for candidate in candidates:
        if is_shared_ffmpeg(candidate):
            return candidate.resolve()
    return None


def install_local_ffmpeg_shared() -> Path:
    """Download, verify SHA256, and extract FFmpeg 7.1.1 full_build-shared into project directory."""
    if ffmpeg_711_is_complete(LOCAL_FFMPEG_BIN):
        say(f"✓ 项目内置 FFmpeg Shared 7.1.1 已就绪：{LOCAL_FFMPEG_BIN}")
        return LOCAL_FFMPEG_BIN

    LOCAL_FFMPEG_BIN.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ffmpeg_setup_", dir=str(LOCAL_FFMPEG_BIN.parent)) as tmp_dir:
        tmp = Path(tmp_dir)
        archive = tmp / "ffmpeg-7.1.1-full_build-shared.zip"
        say("正在下载官方 FFmpeg 7.1.1 Full Shared 组件（用于 TorchCodec 音频解码）……")
        say(f"下载源：{FFMPEG_URL}")
        request = urllib.request.Request(
            FFMPEG_URL,
            headers={"User-Agent": "CoverStudio-Windows-Installer/2.0"},
        )
        hasher = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                hasher.update(chunk)
        actual_hash = hasher.hexdigest().lower()
        if actual_hash != FFMPEG_SHA256:
            raise RuntimeError(
                f"FFmpeg 压缩包 SHA256 校验失败，已拒绝解压。\n期望值: {FFMPEG_SHA256}\n实际值: {actual_hash}"
            )
        say("✓ FFmpeg 7.1.1 压缩包 SHA256 校验通过")

        stage = tmp / "bin"
        stage.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                parts = member.filename.replace("\\", "/").split("/")
                if (
                    member.is_dir()
                    or len(parts) != 3
                    or parts[0] != "ffmpeg-7.1.1-full_build-shared"
                    or parts[1] != "bin"
                ):
                    continue
                name = parts[-1]
                if not name.lower().endswith((".dll", ".exe")):
                    continue
                with zf.open(member) as src, (stage / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1

        if not ffmpeg_711_is_complete(stage):
            raise RuntimeError("解压后的 FFmpeg 缺少必要的 Shared DLL 文件。")

        if LOCAL_FFMPEG_BIN.exists():
            shutil.rmtree(LOCAL_FFMPEG_BIN)
        shutil.move(str(stage), str(LOCAL_FFMPEG_BIN))
        say(f"✓ 已安装 {count} 个 FFmpeg Shared DLL/EXE 文件至：{LOCAL_FFMPEG_BIN}")

    return LOCAL_FFMPEG_BIN


def check_pytorch(py: Path, label: str) -> bool:
    """Verify CUDA 12.6 and GPU tensor execution using ASCII markers."""
    if not py.is_file():
        say(f"⚠ {label} 虚拟环境不存在：{py}")
        return False
    code = (
        "import torch\n"
        "print('TORCH_VERSION=' + str(torch.__version__), flush=True)\n"
        "print('TORCH_CUDA=' + str(torch.version.cuda), flush=True)\n"
        "available = torch.cuda.is_available()\n"
        "print('CUDA_AVAILABLE=' + str(available), flush=True)\n"
        "if available:\n"
        "    print('GPU_NAME=' + torch.cuda.get_device_name(0), flush=True)\n"
        "    value = torch.ones((8,), device='cuda').sum().item()\n"
        "    print('CUDA_TENSOR_SUM=' + str(value), flush=True)\n"
    )
    res = capture([str(py), "-c", code])
    say(f"[{label} 环境检测]\n" + "\n".join("  " + line for line in res.stdout.strip().splitlines()))
    markers: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k in {"TORCH_VERSION", "TORCH_CUDA", "CUDA_AVAILABLE", "CUDA_TENSOR_SUM"}:
                markers[k] = v.strip()
    try:
        value_ok = float(markers.get("CUDA_TENSOR_SUM", "nan")) == 8.0
    except ValueError:
        value_ok = False
    valid = (
        res.returncode == 0
        and markers.get("TORCH_CUDA") == "12.6"
        and markers.get("CUDA_AVAILABLE") == "True"
        and value_ok
    )
    if valid:
        say(f"✓ {label} CUDA 12.6 GPU 张量运算测试通过")
    else:
        say(f"⚠ {label} GPU 检测未通过 (退出码 {res.returncode})")
    return valid


def check_torchcodec(ffmpeg_dir: Path | None = None) -> bool:
    """Test TorchCodec AudioDecoder and torchaudio.load in the tsumugi venv."""
    if not TSUMUGI_PY.is_file():
        say("⚠ tsumugi Python 环境尚未安装，无法测试 TorchCodec。")
        return False
    target_ffmpeg = ffmpeg_dir or ffmpeg_shared_bin()
    if not target_ffmpeg:
        say("⚠ 未找到 FFmpeg Shared DLL 目录，无法测试 TorchCodec。")
        return False

    with tempfile.TemporaryDirectory(prefix="torchcodec_probe_") as tmp_dir:
        wav = Path(tmp_dir) / "silence.wav"
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 2205)

        code = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "ffmpeg_dir = Path(sys.argv[1])\n"
            "wav = sys.argv[2]\n"
            "_ffmpeg_dll_handle = os.add_dll_directory(str(ffmpeg_dir))\n"
            "os.environ['PATH'] = str(ffmpeg_dir) + os.pathsep + os.environ.get('PATH', '')\n"
            "from torchcodec.decoders import AudioDecoder\n"
            "samples = AudioDecoder(wav).get_all_samples()\n"
            "print('TORCHCODEC_SAMPLES=' + str(samples.data.shape), flush=True)\n"
            "import torchaudio\n"
            "audio, sr = torchaudio.load(wav)\n"
            "print('TORCHAUDIO_LOAD=' + str(audio.shape) + ' sr=' + str(sr), flush=True)\n"
            "print('TORCHCODEC_PROBE_OK=1', flush=True)\n"
        )
        env = os.environ.copy()
        env["PATH"] = str(target_ffmpeg) + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        res = capture([str(TSUMUGI_PY), "-X", "utf8", "-c", code, str(target_ffmpeg), str(wav)], env=env)
        ok = res.returncode == 0 and "TORCHCODEC_PROBE_OK=1" in res.stdout
        if ok:
            say("✓ TorchCodec 与 torchaudio 音频解码自检通过")
        else:
            say("⚠ TorchCodec 音频解码自检失败：\n" + res.stdout[-2000:])
        return ok


def fix_ffmpeg() -> None:
    """Standalone command to install/repair FFmpeg 7.1.1 Shared and verify TorchCodec."""
    check_system()
    say("=== 修复 FFmpeg Shared 7.1.1 与 TorchCodec 解码组件 ===")
    bin_dir = install_local_ffmpeg_shared()
    patch_worker_ffmpeg_handle()
    if not check_torchcodec(bin_dir):
        raise RuntimeError("TorchCodec 解码测试未通过，请检查上方错误输出。")
    say("\n✓ FFmpeg Shared 与 TorchCodec 修复并验证完成！")


def configure_performance(preset: int | None = None) -> None:
    """Interactive or CLI selector for transcription window-batch-size."""
    current = get_window_batch_size()
    say("=== Cover Studio 转录显存/性能档位设置 ===")
    say(f"当前档位：batch={current} — {BATCH_OPTIONS.get(current, '')}\n")
    for k, desc in BATCH_OPTIONS.items():
        marker = " (当前)" if k == current else ""
        say(f"  [{k}] {desc}{marker}")

    if preset is None:
        raw = input(f"\n请选择档位 1/2/3/4 (直接回车保持 {current}): ").strip()
        selected = current if not raw else int(raw) if raw.isdigit() else -1
    else:
        selected = preset

    if selected not in BATCH_OPTIONS:
        raise RuntimeError("无效的档位选择，请输入 1、2、3 或 4。")

    BATCH_CONFIG.write_text(f"{selected}\n", encoding="ascii")
    say(f"\n✓ 已保存配置：window-batch-size={selected}")
    if WORKER.is_file():
        patch_worker_batch_size()
    say("提示：如果服务正在运行，请重启服务使其生效。")


def download_models() -> None:
    """Prefetch Hugging Face models with Xet disabled to avoid 0/N download hangs."""
    check_system()
    if not COVER_PY.is_file():
        raise RuntimeError("尚未安装 Cover Studio 虚拟环境，请先运行 01_install.bat。")

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    say("=== 预下载 Hugging Face 模型权重 (禁用 Xet 防卡死模式) ===")
    say(f"缓存目录：{HF_CACHE}")
    say("提示：模型总大小约 3~4 GB，下载期间 GPU 不会工作，请耐心等待。\n")

    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        "repos = sys.argv[1:]\n"
        "for repo_id in repos:\n"
        "    print(f'\\n>>> 正在下载/校验模型: {repo_id} ...', flush=True)\n"
        "    p = Path(snapshot_download(repo_id=repo_id, repo_type='model', max_workers=2, etag_timeout=30))\n"
        "    files = [f for f in p.rglob('*') if f.is_file()]\n"
        "    print(f'✓ {repo_id} 就绪 ({len(files)} 个文件): {p}', flush=True)\n"
    )
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["HF_HUB_ETAG_TIMEOUT"] = "30"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run([str(COVER_PY), "-c", code, *MODEL_REPOS], env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError("模型下载未完成，请检查网络连接后重试（支持断点续传）。")
    say("\n✓ 全部核心模型下载与校验完成！")


def install() -> None:
    check_system()
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    say("============================================================")
    say("  Cover Studio · Windows 10/11 (RTX 3060 6GB) 一键部署安装")
    say("============================================================")
    say(f"安装目录：{BASE}")
    free_gb = shutil.disk_usage(BASE).free / (1024**3)
    say(f"当前磁盘剩余空间：{free_gb:.1f} GiB（建议预留 30 GiB 以上）")
    if free_gb < 15:
        say("⚠ 可用磁盘空间较小，依赖与模型权重可能占满空间，建议先清理磁盘。")

    nvidia_report()
    if not shutil.which("git"):
        raise RuntimeError("未检测到 Git，请先安装 Git for Windows (https://git-scm.com/downloads/win) 并重开终端。")
    uv = get_uv()

    say("\n[1/6] 拉取上游开源仓库源码")
    ensure_clone("https://github.com/anime-song/tsumugi-piano-cover.git", COVER)
    ensure_clone("https://github.com/anime-song/tsumugi.git", TSUMUGI)

    say("\n[2/6] 应用 CUDA 12.6 驱动兼容补丁 (torch 2.13.0+cu126)")
    patch_cuda_126_sources()

    say("\n[3/6] 创建并安装 Cover Studio 虚拟环境")
    if not COVER_PY.is_file():
        call([str(uv), "venv", "--python", sys.executable, str(COVER / ".venv")])
    req = COVER / "requirements.txt"
    if not req.is_file():
        raise RuntimeError(f"缺少依赖文件：{req}")
    call([str(uv), "pip", "install", "--python", str(COVER_PY), "-r", str(req)], cwd=COVER)

    say("\n[4/6] 创建并安装 tsumugi 转录虚拟环境 (重新解析 cu126 锁文件)")
    call([str(uv), "sync", "--extra", "stem", "--python", sys.executable], cwd=TSUMUGI)
    if not TSUMUGI_PY.is_file():
        raise RuntimeError(f"tsumugi 虚拟环境未生成预期的 Python：{TSUMUGI_PY}")

    say("\n[5/6] 部署项目内置 FFmpeg 7.1.1 Shared 与 Worker 补丁")
    patch_worker_batch_size()
    patch_worker_ffmpeg_handle()
    patch_midi_export_fix()
    patch_web_midi_player()
    ffmpeg = ffmpeg_shared_bin()
    if ffmpeg:
        say(f"✓ 检测到可用 FFmpeg Shared DLL：{ffmpeg}")
    else:
        try:
            ffmpeg = install_local_ffmpeg_shared()
        except Exception as exc:
            say(f"⚠ 自动下载 FFmpeg Shared 失败 ({exc})，稍后可通过 03_toolbox.bat 重试。")

    say("\n[6/6] 运行 GPU CUDA 12.6 与 TorchCodec 音频解码自检")
    cuda_cover = check_pytorch(COVER_PY, "Cover Studio")
    cuda_tsumugi = check_pytorch(TSUMUGI_PY, "tsumugi")
    codec_ok = check_torchcodec(ffmpeg)
    if cuda_cover and cuda_tsumugi and codec_ok:
        say("\n🎉 一键部署全部完成！双击 02_start.bat 即可启动 Cover Studio。")
    else:
        say("\n⚠ 安装已完成，但部分自检未通过，请查看上方提示或运行 03_toolbox.bat 排查。")


def is_port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start(mode: str) -> None:
    check_system()
    if not COVER_PY.is_file() or not TSUMUGI_PY.is_file():
        raise RuntimeError("尚未安装运行环境，请先双击运行 01_install.bat。")
    if is_port_open(PORT):
        raise RuntimeError(f"端口 {PORT} 已被占用。若服务已在运行，请直接访问 http://127.0.0.1:{PORT}")

    batch = get_window_batch_size()
    patch_worker_batch_size()
    patch_worker_ffmpeg_handle()
    patch_midi_export_fix()
    patch_web_midi_player()

    if not nvidia_report():
        raise RuntimeError("未检测到 NVIDIA GPU，请检查显卡驱动。")
    if not check_pytorch(COVER_PY, "Cover Studio"):
        raise RuntimeError("Cover Studio 的 CUDA 12.6 环境未通过自检，请运行 03_toolbox.bat 进行诊断。")
    if mode == "gpu" and not check_pytorch(TSUMUGI_PY, "tsumugi"):
        raise RuntimeError("tsumugi 的 GPU 环境未就绪，可尝试切换 CPU 转录模式。")

    ffmpeg = ffmpeg_shared_bin()
    if not ffmpeg:
        say("⚠ 未检测到完整 FFmpeg Shared DLL，正在尝试自动安装……")
        try:
            ffmpeg = install_local_ffmpeg_shared()
        except Exception as exc:
            say(f"⚠ FFmpeg 自动安装未完成 ({exc})；仅 MIDI 模式可用，音频转录可能失败。")

    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COVER_PY),
        "-m",
        "cover_studio",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--root",
        str(DATA),
        "--tsumugi-dir",
        str(TSUMUGI),
        "--tsumugi-python",
        str(TSUMUGI_PY),
        "--pretrained-dir",
        "",
        "--checkpoints-dir",
        "",
        "--device",
        "cuda",
        "--tsumugi-device",
        "cuda" if mode == "gpu" else "cpu",
    ]
    if ffmpeg:
        cmd.extend(["--ffmpeg-dir", str(ffmpeg)])

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HOME"] = str(HF_CACHE)
    env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    # Built-in non-Xet fallback so Hugging Face large model downloads do not hang on Windows
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["HF_HUB_ETAG_TIMEOUT"] = "30"

    log_path = LOGS / "cover_studio.log"
    say("\n============================================================")
    say(f"  ▶ 访问地址：http://127.0.0.1:{PORT}")
    say(f"  ▶ 转录模式：{'GPU 加速 (window-batch-size=' + str(batch) + ')' if mode == 'gpu' else 'CPU 兼容模式 (翻弹生成仍用 GPU)'}")
    say(f"  ▶ 模型缓存：{HF_CACHE} (已启用 HF_HUB_DISABLE_XET=1 防卡死)")
    say(f"  ▶ 项目数据：{DATA}")
    say("  ▶ 按 Ctrl+C 或直接关闭窗口即可停止服务")
    say("============================================================\n")

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=COVER,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 60 + f"\n{datetime.now().isoformat()}\n")
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
        if proc.wait() != 0:
            raise RuntimeError(f"服务异常退出 (退出码 {proc.returncode})，请查看日志：{log_path}")
    except KeyboardInterrupt:
        say("\n正在停止 Cover Studio 服务……")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        say("✓ 服务已停止。")


def check() -> None:
    check_system()
    say("=== Cover Studio 环境与解码全面自检 ===")
    say(f"Python 版本: {sys.version.split()[0]} ({sys.executable})")
    say(f"项目根目录 : {BASE}")
    say(f"Git 路径   : {shutil.which('git') or '未找到'}")
    say(f"uv 路径    : {find_uv() or '未找到'}")
    say(f"当前性能档位: batch={get_window_batch_size()} ({BATCH_OPTIONS.get(get_window_batch_size(), '')})")
    nvidia_report()
    check_pytorch(COVER_PY, "Cover Studio")
    check_pytorch(TSUMUGI_PY, "tsumugi")
    patch_worker_ffmpeg_handle()
    patch_midi_export_fix()
    patch_web_midi_player()
    ffmpeg = ffmpeg_shared_bin()
    say(f"FFmpeg Shared: {ffmpeg or '未找到 (请通过 03_toolbox.bat 执行 FFmpeg 修复)'}")
    if ffmpeg:
        check_torchcodec(ffmpeg)
    say(f"项目数据目录: {DATA}")
    say(f"模型缓存目录: {HF_CACHE}")


def toolbox() -> None:
    """Interactive menu consolidating check, performance tuning, model prefetch, FFmpeg repair, and CPU fallback."""
    check_system()
    while True:
        batch = get_window_batch_size()
        say("\n============================================================")
        say("  Cover Studio · 实用工具箱与故障排查菜单")
        say("============================================================")
        say(f"  当前转录档位: batch={batch} | {BATCH_OPTIONS.get(batch, '')}")
        say("------------------------------------------------------------")
        say("  [1] 环境全面自检 (检测 GPU / CUDA 12.6 / FFmpeg / TorchCodec)")
        say("  [2] 切换显存性能档位 (设置转录 window-batch-size: 1~4)")
        say("  [3] 预下载/修复模型权重 (SheetSage2 & V2，禁用 Xet 防卡死)")
        say("  [4] 下载/修复 FFmpeg Shared 7.1.1 (解决 TorchCodec 解码报错)")
        say("  [5] 启动服务 - CPU 转录兼容模式 (解决 6GB 显存转录 OOM)")
        say("  [0] 退出工具箱")
        say("============================================================")
        try:
            choice = input("请输入选项编号 [0-5]: ").strip()
        except EOFError:
            return
        if choice == "0" or not choice:
            return
        try:
            if choice == "1":
                check()
            elif choice == "2":
                configure_performance()
            elif choice == "3":
                download_models()
            elif choice == "4":
                fix_ffmpeg()
            elif choice == "5":
                start("transcribe-cpu")
                return
            else:
                say("⚠ 请输入有效的编号 (0~5)。")
        except Exception as exc:
            say(f"\n❌ 执行出错：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("install", help="One-click clone, install CUDA 12.6 venvs, FFmpeg Shared, and self-test")
    sub.add_parser("check", help="Check NVIDIA GPU, CUDA 12.6, FFmpeg Shared, and TorchCodec decoding")
    sub.add_parser("fix-ffmpeg", help="Download/verify FFmpeg 7.1.1 Shared and test TorchCodec")
    sub.add_parser("download-models", help="Prefetch Hugging Face models with Xet disabled")
    sub.add_parser("toolbox", help="Open interactive maintenance and troubleshooting menu")

    p_perf = sub.add_parser("performance", help="Configure transcription window-batch-size (1..4)")
    p_perf.add_argument("--set", dest="preset", type=int, choices=(1, 2, 3, 4), default=None)

    p_start = sub.add_parser("start", help="Start Cover Studio local server")
    p_start.add_argument("--mode", choices=("gpu", "transcribe-cpu"), default="gpu")

    args = parser.parse_args()
    try:
        if args.action == "install":
            install()
        elif args.action == "start":
            start(args.mode)
        elif args.action == "check":
            check()
        elif args.action == "fix-ffmpeg":
            fix_ffmpeg()
        elif args.action == "performance":
            configure_performance(args.preset)
        elif args.action == "download-models":
            download_models()
        elif args.action == "toolbox":
            toolbox()
    except Exception as exc:
        say(f"\n❌ {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
