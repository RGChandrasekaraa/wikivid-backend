import argparse
import logging
import os
import sys
import textwrap
import time
import uuid
from functools import partial
from multiprocessing import Pool
from .s3_utils import upload_file_to_s3, bucket_exists

from moviepy.editor import concatenate_videoclips, TextClip, AudioFileClip, CompositeVideoClip, vfx
from gtts import gTTS
from pydub.utils import mediainfo
import wikipediaapi
from transformers import BartTokenizer, pipeline


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for text and video parameters
WRAP_WIDTH = 40
FONT_SIZE = 24
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
INITIAL_DELAY = 2  # seconds
AUDIO_CHUNK_SIZE = 1024


def get_audio_duration(audio_path):
    try:
        audio_info = mediainfo(audio_path)
        return float(audio_info['duration'])
    except Exception as e:
        logging.error(f"Error getting audio duration: {e}")
        return 0


def get_wikipedia_content(url):
    try:
        user_agent = "wikiaudio/0.1 (https://github.com/rgchandrasekaraa/wikivid; rgchandrasekaraa@gmail.com)"
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en', user_agent=user_agent)
        title = url.split("/")[-1].replace("_", " ")
        page = wiki_wiki.page(title)
        if not page.exists():
            logging.error("Page does not exist.")
            return ""
        return page.text
    except Exception as e:
        logging.error(f"Error fetching Wikipedia content: {e}")
        return ""


def summarize_text(text):
    model_name = "facebook/bart-base"  # Using the base BART model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization", model=model_name, tokenizer=tokenizer)
    summarized = ""
    try:
        # Adjust max_length and min_length if needed
        max_length = 200
        min_length = 50

        # Check if the text is too long for one go
        if len(tokenizer.tokenize(text)) > tokenizer.model_max_length:
            logging.warning(
                f"Text is too long, it will be truncated to {tokenizer.model_max_length} tokens for summarization.")
            text_chunks = tokenizer.encode(
                text, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
            summarized = summarizer.generate(
                text_chunks, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            summarized = tokenizer.decode(
                summarized[0], skip_special_tokens=True)
        else:
            summarized = summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False)
            summarized = summarized[0]['summary_text']
        return summarized
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return text


def text_to_audio(text, language='en'):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio_path = f"{uuid.uuid4().hex}.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        logging.error(f"Error converting text to audio: {e}")
        return None


def create_video(text, audio_path, token):
    try:
        audio_duration = get_audio_duration(audio_path)
        wrapped_text = textwrap.fill(text, width=WRAP_WIDTH)
        total_text_height = FONT_SIZE * len(wrapped_text.split('\n'))
        video_height = max(VIDEO_HEIGHT, total_text_height + 100)

        text_clip = TextClip(wrapped_text, fontsize=FONT_SIZE,
                             color='white', size=(VIDEO_WIDTH, video_height))
        scrolling_text_clip = text_clip.set_duration(
            audio_duration - INITIAL_DELAY).fx(vfx.scroll, y_speed=-50)

        static_text_clip = TextClip(wrapped_text, fontsize=FONT_SIZE, color='white', size=(
            VIDEO_WIDTH, video_height)).set_duration(INITIAL_DELAY)
        final_clip = concatenate_videoclips(
            [static_text_clip, scrolling_text_clip])

        audio = AudioFileClip(audio_path)
        video = CompositeVideoClip(
            [final_clip.set_audio(audio).set_position("center")])
        video_path = f"{token}_output_video.mp4"
        video.write_videofile(video_path, codec='libx264',
                              audio_codec='aac', fps=24)
        return video_path
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return None


def generate_video(url: str, token: str):
    logging.info(f"Processing URL: {url}")
    article_text = get_wikipedia_content(url)
    if not article_text:
        logging.error("Failed to retrieve content. Exiting.")
        return

    logging.info("Content fetched. Summarizing...")
    summarized_text = summarize_text(article_text)

    logging.info("Text summarized. Generating audio...")
    audio_path = text_to_audio(summarized_text)
    if not audio_path:
        logging.error("Failed to generate audio. Exiting.")
        return

    logging.info("Audio generated. Creating video...")
    video_path = create_video(summarized_text, audio_path, token)
    if video_path:
        logging.info(f"Video has been saved to: {video_path}")

        # Check if bucket exists and upload to S3
        bucket_name = 'wikivid'  # Specify your bucket name here
        if bucket_exists(bucket_name):
            if upload_file_to_s3(video_path, bucket_name, os.path.basename(video_path)):
                logging.info(
                    f"Video {video_path} uploaded to S3 bucket {bucket_name}.")
                # Remove the local video file after upload
                os.remove(video_path)
            else:
                logging.error(
                    f"Failed to upload video {video_path} to S3 bucket {bucket_name}.")
        else:
            logging.error(f"S3 bucket {bucket_name} does not exist.")

        # Clean up audio file regardless of S3 upload success
        os.remove(audio_path)
    else:
        logging.error("Failed to create video. Exiting.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate videos from Wikipedia content.")
    parser.add_argument('urls', nargs='+',
                        help='List of Wikipedia URLs to process.')
    parser.add_argument('--processes', type=int, default=2,
                        help='Number of parallel processes to use.')
    return parser.parse_args()


def create_scrolling_text(text, audio_duration, width=VIDEO_WIDTH, font_size=FONT_SIZE, initial_delay=INITIAL_DELAY):
    wrapped_text = textwrap.fill(text, width=40)
    text_clip = TextClip(wrapped_text, fontsize=font_size,
                         color='white', size=(width, 'auto'))

    # Calculate the scrolling speed based on text length and audio duration
    scrolling_speed = (text_clip.h - VIDEO_HEIGHT) / \
        (audio_duration - initial_delay)
    scrolling_text_clip = text_clip.set_duration(
        audio_duration - initial_delay).fx(vfx.scroll, y_speed=scrolling_speed)

    static_text_clip = TextClip(wrapped_text, fontsize=font_size, color='white', size=(
        width, 'auto')).set_duration(initial_delay)
    final_clip = concatenate_videoclips(
        [static_text_clip, scrolling_text_clip])

    return final_clip


def run_in_parallel(urls, num_processes):
    tokens = [str(uuid.uuid4()) for _ in urls]  # Generate unique tokens
    pool = Pool(processes=num_processes)
    pool.starmap(generate_video, zip(urls, tokens))
    pool.close()
    pool.join()


def process_url(url, token):
    generate_video(url, token)


def run_concurrently(urls):
    processes = []

    # Create a process for each URL
    for url in urls:
        token = str(uuid.uuid4())
        proc = Process(target=process_url, args=(url, token))
        processes.append(proc)
        proc.start()

    # Wait for all processes to finish
    for proc in processes:
        proc.join()


def main():
    args = parse_arguments()
    run_concurrently(args.urls)


if __name__ == "__main__":
    main()
