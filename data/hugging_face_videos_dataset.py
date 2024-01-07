import os
import time

import torch
import scrapetube
from pytube import YouTube
from faster_whisper import WhisperModel
from tqdm import tqdm
    

if torch.cuda.is_available():
    # requires: conda install -c anaconda cudnn
    print("Using GPU and float16")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", device_index=[1])
else:
    print("Using CPU and int8")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")


def get_videos_urls(channel_url: str) -> list[str]:
    videos = scrapetube.get_channel(channel_url=channel_url)
    return [
        f"https://www.youtube.com/watch?v={video['videoId']}"
        for video in videos
    ]


def get_audio_from_video(video_url: str, save_path: str) -> tuple[str, int, str, int]:
    yt = YouTube(video_url) 
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=save_path) 
    base, ext = os.path.splitext(out_file)
    new_file = base.replace(" ", "_") + ".mp3"
    os.rename(out_file, new_file)
    print(f'Video length: {yt.length} seconds')
    return (video_url, new_file, yt.title, yt.length)


def check_if_transcript_exists(video_name: str, transcripts_save_path: str) -> bool:
    title = video_name.replace(' ', '_')
    return any([
        title in filename
        for filename in os.listdir(transcripts_save_path)
    ])


def transcript_from_audio(audio_path: str) -> dict[str, list[str]]:
    segments, info = model.transcribe(audio_path, beam_size=10)
    return list(segments)


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
            merged_segments[key] = temp_text.strip()
            temp_text = ''

    return merged_segments


def main():
    audio_save_path = 'datasets/huggingface_audio/'
    transcripts_save_path = 'datasets/huggingface_audio_transcribed/'
    if not os.path.exists(audio_save_path):
        os.makedirs(audio_save_path)
    if not os.path.exists(transcripts_save_path):
        os.makedirs(transcripts_save_path)

    print('Getting videos urls')
    videos_urls = get_videos_urls('https://www.youtube.com/@HuggingFace')

    print('Downloading audio files')
    audio_data = []
    for video_url in tqdm(videos_urls):
        try:
            audio_data.append(
                get_audio_from_video(video_url, save_path=audio_save_path)
            )
        except Exception as e:
            print(f'Error downloading video: {video_url}')
            print(e)

    print('Transcribing audio files')
    for video_url, filename, title, audio_length in tqdm(audio_data):
        if check_if_transcript_exists(title, transcripts_save_path):
            print(f'Transcript already exists for: {title}')
            continue
        print(f'Transcribing: {title}')
        start_time = time.time()
        segments = transcript_from_audio(filename)
        print(f'Transcription took {time.time() - start_time} seconds')
        merged_segments = merge_transcripts_segements(
            segments,
            title,
            num_segments_to_merge=10
        )
        # save transcripts to separate files
        title = title.replace(' ', '_')
        for segment, text in merged_segments.items():
            with open(f'{transcripts_save_path}{title}_{segment}.txt', 'w') as f:
                video_url_with_time = f'{video_url}&t={float(segment.split("_")[0]):.0f}'
                f.write(f'source: {video_url_with_time}\n\n' + text)


if __name__ == '__main__':
    main()
