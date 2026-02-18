#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DI_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
MESH_BUILD_SCRIPT="$DI_ROOT/mesh-llm/scripts/build-mesh-binary.sh"
LLAMA_BIN_DIR="$DI_ROOT/llama.cpp/build/bin"
RPC_SERVER_PATH="$LLAMA_BIN_DIR/rpc-server"
LLAMA_SERVER_PATH="$LLAMA_BIN_DIR/llama-server"
APP_ICON_DIR="$ROOT_DIR/Sources/Resources"
APP_ICON_PATH="$APP_ICON_DIR/MeshLLM.icns"
DIST_DIR="$ROOT_DIR/dist"
APP_EXECUTABLE="MeshLLMMenuBar"
APP_DISPLAY_NAME="Mesh LLM"
APP_BUNDLE="$DIST_DIR/$APP_DISPLAY_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

if [ ! -x "$MESH_BUILD_SCRIPT" ]; then
  echo "Missing executable mesh build script: $MESH_BUILD_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$APP_ICON_PATH" ]; then
  echo "Missing app icon: $APP_ICON_PATH" >&2
  exit 1
fi

echo "[1/3] Building mesh-llm from source tree"
BUILD_OUTPUT="$($MESH_BUILD_SCRIPT)"
MESH_BINARY_PATH="$(printf '%s\n' "$BUILD_OUTPUT" | awk -F= '/^BINARY_PATH=/{print $2}' | tail -n1)"

if [ -z "$MESH_BINARY_PATH" ] || [ ! -x "$MESH_BINARY_PATH" ]; then
  echo "Mesh build did not produce a valid BINARY_PATH" >&2
  printf '%s\n' "$BUILD_OUTPUT" >&2
  exit 1
fi

echo "[2/4] Ensuring llama.cpp runtime binaries (rpc-server, llama-server)"
if [ ! -x "$RPC_SERVER_PATH" ] || [ ! -x "$LLAMA_SERVER_PATH" ]; then
  if ! command -v just >/dev/null 2>&1; then
    echo "Missing required binaries and 'just' is not installed." >&2
    echo "Install 'just' and run: (cd \"$DI_ROOT\" && just build)" >&2
    exit 1
  fi
  echo "Building distributed-inference prerequisites via just build..."
  (cd "$DI_ROOT" && just build)
fi

if [ ! -x "$RPC_SERVER_PATH" ] || [ ! -x "$LLAMA_SERVER_PATH" ]; then
  echo "Required binaries not found after build:" >&2
  echo "  $RPC_SERVER_PATH" >&2
  echo "  $LLAMA_SERVER_PATH" >&2
  exit 1
fi

echo "[3/4] Building release binary"
cd "$ROOT_DIR"
if pgrep -x "$APP_EXECUTABLE" >/dev/null 2>&1; then
  echo "Stopping running $APP_EXECUTABLE process..."
  pkill -x "$APP_EXECUTABLE" || true
  sleep 1
fi
echo "Cleaning Swift package build artifacts..."
swift package clean
swift build -c release

BIN_PATH="$ROOT_DIR/.build/release/$APP_EXECUTABLE"
if [ ! -x "$BIN_PATH" ]; then
  echo "Missing built binary: $BIN_PATH" >&2
  exit 1
fi

echo "[4/4] Assembling app bundle"
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp "$BIN_PATH" "$MACOS_DIR/$APP_EXECUTABLE"
cp "$MESH_BINARY_PATH" "$RESOURCES_DIR/mesh-llm"
chmod +x "$RESOURCES_DIR/mesh-llm"
cp "$RPC_SERVER_PATH" "$RESOURCES_DIR/rpc-server"
cp "$LLAMA_SERVER_PATH" "$RESOURCES_DIR/llama-server"
chmod +x "$RESOURCES_DIR/rpc-server" "$RESOURCES_DIR/llama-server"
for dylib in "$LLAMA_BIN_DIR"/*.dylib; do
  [ -f "$dylib" ] || continue
  cp "$dylib" "$RESOURCES_DIR/"
done
cp "$APP_ICON_PATH" "$RESOURCES_DIR/MeshLLM.icns"
for icon in "$APP_ICON_DIR"/MeshLLM-*.icns; do
  [ -f "$icon" ] || continue
  cp "$icon" "$RESOURCES_DIR/"
done
for icon in "$APP_ICON_DIR"/MeshLLM-*.png; do
  [ -f "$icon" ] || continue
  cp "$icon" "$RESOURCES_DIR/"
done
RESOURCE_BUNDLE_PATH="$ROOT_DIR/.build/release/${APP_EXECUTABLE}_${APP_EXECUTABLE}.resources"
if [ -d "$RESOURCE_BUNDLE_PATH" ]; then
  cp -R "$RESOURCE_BUNDLE_PATH" "$RESOURCES_DIR/"
fi

cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_EXECUTABLE</string>
  <key>CFBundleIdentifier</key>
  <string>com.meshllm.menubar</string>
  <key>CFBundleName</key>
  <string>$APP_DISPLAY_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleIconFile</key>
  <string>MeshLLM</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>$(date +%s)</string>
  <key>LSUIElement</key>
  <true/>
</dict>
</plist>
PLIST

echo "Built app bundle: $APP_BUNDLE"
echo "Bundled mesh-llm binary: $RESOURCES_DIR/mesh-llm"
echo "Bundled rpc-server: $RESOURCES_DIR/rpc-server"
echo "Bundled llama-server: $RESOURCES_DIR/llama-server"
echo "Bundled app icon: $RESOURCES_DIR/MeshLLM.icns"
echo "Bundled variant icons (if present): $RESOURCES_DIR/MeshLLM-*.{icns,png}"
