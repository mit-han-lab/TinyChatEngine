#!/bin/bash

# Clone whisper.cpp and checkout the specific commit
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
git checkout a4bb2df

# Determine the platform
OS="$(uname)"
if [ "$OS" = "Linux" ]; then
    # Install SDL2 on Linux
    sudo apt-get install libsdl2-dev
elif [ "$OS" = "Darwin" ]; then
    # Install SDL2 on Mac OS
    brew install sdl2
else
    echo "Unsupported operating system: $OS"
    exit 1
fi

# Apply patch and download model
git apply ../application/sts_utils/clean_up.patch
bash ./models/download-ggml-model.sh base.en

# Check for NVIDIA GPU
if lspci | grep -i nvidia > /dev/null; then
    # Compile with CUDA support
    WHISPER_CUBLAS=1 make -j stream
else
    # Compile without CUDA support
    make -j stream
fi

# Set up TTS
cd ../
mkdir TTS
cd TTS
wget "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz"
tar -xvzf piper_arm64.tar.gz
rm piper_arm64.tar.gz

# Download default voice
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" -O en_US-amy-medium.onnx
wget "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" -O en_US-amy-medium.onnx.json

# Return to the parent directory and compile chat
cd ../
make clean
make -j chat

echo ""
echo "TinyChatEngine's speech-to-speech chatbot setup completed successfully!"
echo "Use './chat -v' on Linux/MacOS or 'chat.exe -v' on Windows."
echo ""
