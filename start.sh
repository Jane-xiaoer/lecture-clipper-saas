#!/bin/bash
# 启动 lecture-clipper SaaS（本地开发）
set -e

cd "$(dirname "$0")"

echo "=== lecture-clipper SaaS ==="

# 检查依赖
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "安装依赖..."
  pip3 install -r requirements.txt
fi

# 检查 ffmpeg
if ! python3 app/setup_ffmpeg.py 2>/dev/null | grep -q "可用"; then
  echo "配置 ffmpeg..."
  python3 app/setup_ffmpeg.py
fi

echo ""
echo "启动服务: http://localhost:8000"
echo "按 Ctrl+C 停止"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
