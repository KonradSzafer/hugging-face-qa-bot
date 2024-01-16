import os
import re
import time

import torch
import scrapetube
from pytube import YouTube
from faster_whisper import WhisperModel
from tqdm import tqdm


# Available models:
# tiny.en, tiny, base.en, base, small.en, small, medium.en, medium
# large-v1, large-v2, large-v3, large
MODEL_NAME = "large-v3"
AUDIO_SAVE_PATH = 'datasets/huggingface_audio/'
TRANSCRIPTS_SAVE_PATH = 'datasets/huggingface_audio_transcribed/'

if torch.cuda.is_available():
    # requires: conda install -c anaconda cudnn
    print(f"Using {MODEL_NAME} on GPU and float16")
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16", device_index=[5])
else:
    print(f"Using {MODEL_NAME} on CPU and int8")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def replace_unallowed_chars(filename: str) -> str:
    unallowed_chars = [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in unallowed_chars:
        filename = filename.replace(char, '_')
    return filename


def get_videos_urls(channel_url: str) -> list[str]:
    videos = scrapetube.get_channel(channel_url=channel_url)
    return [
        f"https://www.youtube.com/watch?v={video['videoId']}"
        for video in videos
    ]


def get_audio_from_video(video_url: str, save_path: str) -> tuple[str, int, str, int]:
    yt = YouTube(video_url)
    if check_if_file_exists(yt.title, save_path):
        print(f'Audio already exists for: {yt.title}')
        return (video_url, yt.title.replace(" ", "_")+".mp3", yt.title, yt.length)
    else:
        print(f'Downloading audio for: {yt.title}')
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=save_path) 
        base, ext = os.path.splitext(out_file)
        new_filename = save_path + replace_unallowed_chars(yt.title) + '.mp3'
        print(f'Saving audio to: {new_filename}')
        os.rename(out_file, new_filename)
        print(f'Video length: {yt.length} seconds')
        return (video_url, new_filename, yt.title, yt.length)


def check_if_file_exists(filename: str, save_path: str) -> bool:
    title = filename.replace(' ', '_')
    return any([
        title in filename_
        for filename_ in os.listdir(save_path)
    ])


def transcript_from_audio(audio_path: str) -> dict[str, list[str]]:
    segments, info = model.transcribe(audio_path, beam_size=10)
    return list(segments)


def process_text(text: str) -> str:
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text


def merge_transcripts_segements(
    segments: list[str],
    file_title: str,
    num_segments_to_merge: int = 5,
    ) -> dict[str, list[str]]:

    merged_segments = {}
    temp_text = ''
    start_time = None
    end_time = None

    for i, segment in enumerate(segments):
        if i % num_segments_to_merge == 0:
            start_time = segment.start
        end_time = segment.end
        temp_text += segment.text + ' '

        if (i + 1) % num_segments_to_merge == 0 or i == len(segments) - 1:
            key = f'{start_time:.2f}_{end_time:.2f}'
            merged_segments[key] = process_text(temp_text)
            temp_text = ''

    return merged_segments


def main():
    if not os.path.exists(AUDIO_SAVE_PATH):
        os.makedirs(AUDIO_SAVE_PATH)
    if not os.path.exists(TRANSCRIPTS_SAVE_PATH):
        os.makedirs(TRANSCRIPTS_SAVE_PATH)

    print('Getting videos urls')
    videos_urls = get_videos_urls('https://www.youtube.com/@HuggingFace')

    print('Downloading audio files')
    audio_data = []
    for video_url in tqdm(videos_urls):
        try:
            audio_data.append(
                get_audio_from_video(video_url, save_path=AUDIO_SAVE_PATH)
            )
        except Exception as e:
            print(f'Error downloading video: {video_url}')
            print(e)

    print('Transcribing audio files')
    for video_url, filename, title, audio_length in tqdm(audio_data):
        if check_if_file_exists(title, TRANSCRIPTS_SAVE_PATH):
            print(f'Transcript already exists for: {title}')
            continue
        try: 
            print(f'Transcribing: {title}')
            start_time = time.time()
            segments = transcript_from_audio(filename)
            print(f'Transcription took: {time.time() - start_time:.1f} seconds')
            merged_segments = merge_transcripts_segements(
                segments,
                title,
                num_segments_to_merge=10
            )
            # save transcripts to separate files
            title = replace_unallowed_chars(title)
            for segment, text in merged_segments.items():
                with open(f'{TRANSCRIPTS_SAVE_PATH}{title}_{segment}.txt', 'w') as f:
                    video_url_with_time = f'{video_url}&t={float(segment.split("_")[0]):.0f}'
                    f.write(f'source: {video_url_with_time}\n\n' + text)
        except Exception as e:
            print(f'Error transcribing: {title}')
            print(e)


if __name__ == '__main__':
    main()
