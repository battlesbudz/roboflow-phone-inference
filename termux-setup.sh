#!/data/data/com.termux/files/usr/bin/bash
# Jarvis lean inference server - Termux bootstrap.
# Installs everything and launches the server on localhost:9001.
# Run once:  bash termux-setup.sh
# Then start the server anytime:  bash ~/jarvis-vision/run.sh

set -e

echo "==> Updating Termux packages..."
pkg update -y
pkg install -y python python-numpy libjpeg-turbo

echo "==> Creating virtualenv (uses system numpy)..."
python -m venv --system-site-packages ~/jarvis-vision-venv
source ~/jarvis-vision-venv/bin/activate
pip install --upgrade pip

echo "==> Installing Python deps..."
pip install pillow fastapi uvicorn pydantic

echo "==> Installing ONNX Runtime..."
if ! pip install onnxruntime; then
    echo "!! pip install onnxruntime failed. Trying Termux system package..."
    pkg install -y onnxruntime
    # system package puts the python bindings where pip can see them via system-site-packages
fi
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"

echo "==> Setting up server files..."
mkdir -p ~/jarvis-vision
cd ~/jarvis-vision
# server.py and the model/label files go here (copy from your computer or
# download from the repo releases page)

for f in yolov8n.onnx yolov8n-seg.onnx yolov8n-pose.onnx mobilenetv2-12.onnx synset.txt; do
  if [ ! -f "$f" ]; then
    echo ""
    echo "!! $f is missing from ~/jarvis-vision/"
    echo "   Download it from the repo (or the link Scout gave you) and place it here,"
    echo "   then re-run this script."
    exit 1
  fi
done

cat > run.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
source ~/jarvis-vision-venv/bin/activate
cd ~/jarvis-vision
exec python -m uvicorn server:app --host 127.0.0.1 --port 9001
EOF
chmod +x run.sh

echo ""
echo "==> Done. Start the server with:  bash ~/jarvis-vision/run.sh"
echo "    Then check:  curl http://localhost:9001/health"
