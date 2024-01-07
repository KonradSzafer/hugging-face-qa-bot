import os
import time

import scrapetube
from pytube import YouTube
from faster_whisper import WhisperModel
from tqdm import tqdm
    

model = WhisperModel("large-v2")


def get_videos_urls(channel_url: str) -> list[str]:
    videos = scrapetube.get_channel(channel_url=channel_url)
    return [
        f"https://www.youtube.com/watch?v={video['videoId']}"
        for video in videos
    ]


def get_audio_from_video(video_url: str, save_path: str) -> tuple[str, int]:
    yt = YouTube(video_url) 
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=save_path) 
    base, ext = os.path.splitext(out_file)
    new_file = base.replace(" ", "_") + ".mp3"
    os.rename(out_file, new_file)
    print(f'Video length: {yt.length} seconds')
    return (video_url, new_file, yt.title, yt.length)


def transcript_from_audio(audio_path: str) -> dict[str, list[str]]:
    segments, info = model.transcribe(audio_path)
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
    transcripts_save_path = 'datasets/transcripts/'

    print('Getting videos urls')
    videos_urls = get_videos_urls('https://www.youtube.com/@HuggingFace')
    videos_urls = videos_urls[:2]

    print('Downloading audio files')
    audio_paths = []
    for video_url in tqdm(videos_urls):
        audio_paths.append((
            video_url,
            get_audio_from_video(video_url, save_path=audio_save_path)
        ))

    print('Transcribing audio files')
    for video_url, audio_path in tqdm(audio_paths):
        video_title = audio_path.split("/")[-1]
        start_time = time.time()
        print(f'Transcribing: {video_title}')
        segments = transcript_from_audio(audio_path)
        print(f'Transcription took {time.time() - start_time} seconds')
        print(segments)
        merged_segments = merge_transcripts_segements(segments, video_title)
        print(merged_segments)
        # todo: save segments as separate files


if __name__ == '__main__':
    main()
