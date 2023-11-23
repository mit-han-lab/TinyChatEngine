## Demo video of our speech-to-speech chatbot

- Please find the speech-to-speech demo video using TinyChatEngine [here](https://youtu.be/Bw5Dm3aWMnA?si=CCvZDmq3HwowEQcC).

## Instructions to run a speech-to-speech chatbot demo

- Follow the [instructions](../../README.md) to download and deploy LLaMA2-7B-chat.

- Configure whisper.cpp

  ```bash
  # Get whisper.cpp for speech recognition
  cd llm
  git clone https://github.com/ggerganov/whisper.cpp
  cd whisper.cpp
  git checkout a4bb2df

  # Install SDL2 on Linux
  sudo apt-get install libsdl2-dev
  # Install SDL2 on Mac OS
  brew install sdl2

  git apply ../application/sts_utils/clean_up.patch
  bash ./models/download-ggml-model.sh base.en
  # NVIDIA GPU (Note: you may need to change the Makefile of whisper.cpp depending on your environment or device)
  WHISPER_CUBLAS=1 make -j stream
  # Otherwise
  make stream
  cd ../
  ```

- If you have an edge device and want a better TTS program than espeak, download [piper](https://github.com/rhasspy/piper)

  ```bash
    mkdir TTS
    wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
    tar -xvzf piper_arm64.tar.gz
  ```

  - Download your preferred voice from the [huggingface repo](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0) and drag both the .onxx and .onnx.json files into the TTS directory

- Edit the listen shell file in the transformers directory so whisper.cpp is using your preferred parameters.

  ```bash
  nano application/sts_utils/listen
  ```

- Edit the speak shell file in the transformers directory so the demo uses your preferred TTS program.

  ```bash
  nano application/sts_utils/speak
  ```

- Compile and start the voicechat locally.

  ```bash
  make -j chat
  ./chat -v # chat.exe -v on Windows
  ```
