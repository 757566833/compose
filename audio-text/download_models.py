from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def download():
    print("Pre-downloading models...")
    model_dir = "FunAudioLLM/SenseVoiceSmall"
    # 下载 Whisper 模型
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu",
        hub="hf",
    )

if __name__ == "__main__":
    download()