"""
lecture-clipper SaaS — FastAPI 后端

任务状态流：
  queued → transcribing → tagging → review → cutting → postprocessing → done | error
"""
import json, os, shutil, sys, uuid, zipfile
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="lecture-clipper")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS_DIR = Path(__file__).parent.parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────────────

def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def read_status(job_id: str) -> dict:
    f = job_dir(job_id) / "status.json"
    if not f.exists():
        return {"status": "not_found"}
    return json.loads(f.read_text())

def write_status(job_id: str, status: str, progress: int, message: str, extra: dict = {}):
    data = {"job_id": job_id, "status": status, "progress": progress, "message": message, **extra}
    (job_dir(job_id) / "status.json").write_text(json.dumps(data, ensure_ascii=False))

# ── SRT 辅助 ─────────────────────────────────────────────────────

def _parse_srt_times(srt_path) -> list:
    """返回每行的 start/end 秒数列表，与 tagger 行号对齐"""
    import re
    content = Path(srt_path).read_text(encoding="utf-8")
    entries = []
    for block in re.split(r'\n\n+', content.strip()):
        lines = block.strip().split('\n')
        time_line = next((l for l in lines if '-->' in l), None)
        if not time_line:
            continue
        s, e = time_line.split('-->')
        def ts2sec(ts):
            ts = ts.strip().replace(',', '.')
            h, m, sec = ts.split(':')
            return int(h)*3600 + int(m)*60 + float(sec)
        text = ' '.join(l for l in lines if '-->' not in l and not l.strip().isdigit()).strip()
        if text:
            entries.append({"start": ts2sec(s), "end": ts2sec(e), "text": text})
    return entries

def _fmt_min(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"

def _enrich_topics(topics: list, entries: list) -> list:
    """给每个 topic 加上 time_desc（时间段文字）和 preview（首句预览）"""
    n = len(entries)
    result = []
    for t in topics:
        if t.get("id") == "skip":
            continue
        ranges = t.get("ranges", [])
        total_sec = 0.0
        time_parts = []
        preview = ""
        for r in ranges:
            i0 = max(0, r[0])
            i1 = min(n - 1, r[1])
            if i0 >= n:
                continue
            seg_start = entries[i0]["start"]
            seg_end   = entries[i1]["end"]
            total_sec += seg_end - seg_start
            time_parts.append(f"{_fmt_min(seg_start)}–{_fmt_min(seg_end)}")
            if not preview:
                preview = entries[i0]["text"][:40]
        t_copy = dict(t)
        t_copy["time_desc"] = "  |  ".join(time_parts) if time_parts else ""
        t_copy["duration_min"] = round(total_sec / 60, 1)
        t_copy["preview"] = preview
        result.append(t_copy)
    return result

# ── 核心 Pipeline ─────────────────────────────────────────────────

def run_pipeline(job_id: str, api_key: str, provider: str, model_hint: str,
                 whisper_provider: str, whisper_key: str, has_srt: bool):
    """完整 pipeline：Step 0（转写）→ Step 1（标注）→ review 等待用户确认"""
    import subprocess
    d = job_dir(job_id)

    # 设置 LLM API Key 环境变量
    env = os.environ.copy()
    llm_key_map = {
        "gemini":     "GEMINI_API_KEY",
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env[llm_key_map.get(provider, "GEMINI_API_KEY")] = api_key

    try:
        # ── Step 0: 转写（若没有上传 SRT）─────────────────────────
        if not has_srt:
            write_status(job_id, "transcribing", 8, "正在转写音频，请稍候（约 1-5 分钟）...")

            # Whisper provider key
            whisper_env = env.copy()
            if whisper_key:
                if whisper_provider == "groq":
                    whisper_env["GROQ_API_KEY"] = whisper_key
                elif whisper_provider == "openai":
                    whisper_env["OPENAI_API_KEY"] = whisper_key

            r = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "step0_transcribe.py"),
                 "--video",    str(d / "input.mp4"),
                 "--out",      str(d / "input.srt"),
                 "--provider", whisper_provider or "auto"],
                capture_output=True, text=True, env=whisper_env,
                cwd=str(Path(__file__).parent)
            )
            if r.returncode != 0:
                raise RuntimeError(f"转写失败: {r.stderr[-500:]}")

        # ── Step 1: 话题标注 ─────────────────────────────────────
        write_status(job_id, "tagging", 20, "LLM 正在分析话题，约 1-3 分钟...")

        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "step1_tagger.py"),
             "--srt", str(d / "input.srt"),
             "--out", str(d / "metadata"),
             "--model", model_hint or ""],
            capture_output=True, text=True, env=env, cwd=str(Path(__file__).parent)
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])

        # 读取标注结果，enriched with time ranges
        result = json.loads((d / "metadata" / "tagger_result.json").read_text())
        entries = _parse_srt_times(d / "input.srt")
        topics = _enrich_topics(result["topics"], entries)

        write_status(job_id, "review", 35, f"标注完成，共 {len(topics)} 个话题，请确认",
                     extra={"topics": topics})
    except Exception as e:
        write_status(job_id, "error", 0, str(e)[:300])


def run_cutting_and_postprocess(job_id: str):
    """Step 2+3：切片 + 字幕烧入。从 status review 触发。"""
    import subprocess
    d = job_dir(job_id)

    try:
        write_status(job_id, "cutting", 40, "正在切片，请稍候...")
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "step2_cutter.py"),
             "--video", str(d / "input.mp4"),
             "--srt",   str(d / "input.srt"),
             "--meta",  str(d / "metadata"),
             "--out",   str(d / "clips")],
            capture_output=True, text=True, cwd=str(Path(__file__).parent)
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])

        write_status(job_id, "postprocessing", 75, "正在烧入字幕，请稍候...")
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "step3_postprocess.py"),
             "--clips", str(d / "clips"),
             "--out",   str(d / "published")],
            capture_output=True, text=True, cwd=str(Path(__file__).parent)
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])

        # 打包 ZIP
        write_status(job_id, "postprocessing", 95, "打包中...")
        zip_path = d / "output.zip"
        published = list((d / "published").glob("*.mp4"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in published:
                z.write(f, f.name)

        write_status(job_id, "done", 100, f"完成！共 {len(published)} 个视频",
                     extra={"clip_count": len(published)})
    except Exception as e:
        write_status(job_id, "error", 0, f"处理失败: {str(e)[:200]}")

# ── API 路由 ──────────────────────────────────────────────────────

@app.post("/api/submit")
async def submit(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    srt:   Optional[UploadFile] = File(default=None),  # 可选；没有则自动转写
    api_key:          str = Form(...),
    provider:         str = Form(default="gemini"),     # LLM: gemini | openai | anthropic | openrouter
    model:            str = Form(default=""),
    whisper_provider: str = Form(default="auto"),       # 转写: auto | groq | openai | local
    whisper_key:      str = Form(default=""),           # Groq/OpenAI key（可复用 api_key）
):
    job_id = uuid.uuid4().hex[:8]
    d = job_dir(job_id)
    d.mkdir()

    # 保存视频（流式写入，避免内存溢出）
    with open(d / "input.mp4", "wb") as f:
        shutil.copyfileobj(video.file, f)

    has_srt = False
    if srt and srt.filename:
        with open(d / "input.srt", "wb") as f:
            shutil.copyfileobj(srt.file, f)
        has_srt = True

    # 保存任务元数据（retag 时复用）
    (d / "job_meta.json").write_text(json.dumps(
        {"api_key": api_key, "provider": provider, "model": model}, ensure_ascii=False
    ))

    # 若未提供 whisper_key，尝试复用 api_key（OpenAI 用户两个 key 相同）
    effective_whisper_key = whisper_key or (api_key if provider == "openai" else "")

    write_status(job_id, "queued", 3, "已接收，准备处理...")
    background_tasks.add_task(run_pipeline, job_id, api_key, provider, model,
                              whisper_provider, effective_whisper_key, has_srt)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    s = read_status(job_id)
    if s["status"] == "not_found":
        raise HTTPException(404, "任务不存在")
    return s


@app.post("/api/confirm/{job_id}")
async def confirm(job_id: str, background_tasks: BackgroundTasks, body: dict = {}):
    """用户在前端确认话题后，触发切片+烧入"""
    s = read_status(job_id)
    if s.get("status") != "review":
        raise HTTPException(400, f"任务状态不是 review（当前: {s.get('status')}）")

    # 如果用户编辑了话题，保存修改
    if "topics" in body:
        result_path = job_dir(job_id) / "metadata" / "tagger_result.json"
        result = json.loads(result_path.read_text())
        result["topics"] = body["topics"]
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    write_status(job_id, "cutting", 35, "用户已确认，开始切片...")
    background_tasks.add_task(run_cutting_and_postprocess, job_id)
    return {"ok": True}


@app.post("/api/retag/{job_id}")
async def retag(job_id: str, background_tasks: BackgroundTasks):
    """用户觉得话题分组不准，重新跑 LLM 标注"""
    s = read_status(job_id)
    if s.get("status") not in ("review", "error"):
        raise HTTPException(400, f"当前状态不可重新分析（{s.get('status')}）")

    # 读取原始 api_key / provider（保存在 job 目录）
    meta_file = job_dir(job_id) / "job_meta.json"
    if not meta_file.exists():
        raise HTTPException(400, "缺少任务元数据，无法重新分析")
    meta = json.loads(meta_file.read_text())

    write_status(job_id, "tagging", 15, "重新分析话题中，请稍候...")
    background_tasks.add_task(
        _run_tagging_only, job_id, meta["api_key"], meta["provider"], meta.get("model", "")
    )
    return {"ok": True}


def _run_tagging_only(job_id: str, api_key: str, provider: str, model_hint: str):
    """只重跑 Step 1 标注，SRT 已存在"""
    import subprocess
    d = job_dir(job_id)
    env = os.environ.copy()
    llm_key_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                   "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    env[llm_key_map.get(provider, "GEMINI_API_KEY")] = api_key
    try:
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "step1_tagger.py"),
             "--srt", str(d / "input.srt"),
             "--out", str(d / "metadata"),
             "--model", model_hint or ""],
            capture_output=True, text=True, env=env, cwd=str(Path(__file__).parent)
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
        result = json.loads((d / "metadata" / "tagger_result.json").read_text())
        entries = _parse_srt_times(d / "input.srt")
        topics = _enrich_topics(result["topics"], entries)
        write_status(job_id, "review", 35, f"重新分析完成，共 {len(topics)} 个话题",
                     extra={"topics": topics})
    except Exception as e:
        write_status(job_id, "error", 0, str(e)[:300])


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    zip_path = job_dir(job_id) / "output.zip"
    if not zip_path.exists():
        raise HTTPException(404, "文件未就绪")
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"clips_{job_id}.zip")


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """清理任务文件（节省磁盘）"""
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d)
    return {"ok": True}

# ── 前端静态文件 ──────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(Path(__file__).parent.parent / "frontend"),
                           html=True), name="frontend")
