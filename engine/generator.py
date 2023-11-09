import argparse
import logging
import os
import sys
import textwrap
import time
import uuid
from functools import partial
from multiprocessing import Pool
from .s3_utils import upload_fileobj_to_s3
from .db_operations import create_request_entry, finalize_request, update_request_status
from botocore.exceptions import ClientError
import boto3


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


def get_wikipedia_content(token, url):
    try:
        update_request_status(token, 'Fetching Wikipedia content...')
        user_agent = "wikiaudio/0.1 (https://github.com/rgchandrasekaraa/wikivid; rgchandrasekaraa@gmail.com)"
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en', user_agent=user_agent)
        title = url.split("/")[-1].replace("_", " ")
        page = wiki_wiki.page(title)
        if not page.exists():
            logging.error("Page does not exist.")
            return ""
        update_request_status(token, 'Wikipedia content fetched.')
        return page.text
    except Exception as e:
        logging.error(f"Error fetching Wikipedia content: {e}")
        update_request_status(
            token, 'FAILED: While Fetching Wikipedia content', str(e))
        return ""


def summarize_text(token, text):
    model_name = "facebook/bart-base"  # Using the base BART model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization", model=model_name, tokenizer=tokenizer)
    summarized = ""
    try:
        update_request_status(token, 'Summarizing text...')
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
        update_request_status(token, 'Text summarized.')
        return summarized
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        update_request_status(token, 'FAILED: While Summarising Text.', str(e))
        return text


def text_to_audio(token, text, language='en'):
    try:
        update_request_status(token, 'Generating Audio...')
        tts = gTTS(text=text, lang=language, slow=False)
        audio_path = f"{uuid.uuid4().hex}.mp3"
        tts.save(audio_path)
        update_request_status(token, 'Audio Generated.')
        return audio_path
    except Exception as e:
        logging.error(f"Error converting text to audio: {e}")
        update_request_status(token, 'FAILED: While Generating Audio.', str(e))
        return None


def create_video(text, audio_path, token):
    try:
        update_request_status(token, 'Creating video...')
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
        update_request_status(token, 'Video created.')
        return video_path
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        update_request_status(token, 'FAILED: While Creating Video.', str(e))
        return None


def generate_video(url: str, token: str):
    logging.info(f"Processing URL: {url}")

    # Create an entry in DynamoDB when the request is initiated
    create_request_entry(token, url)

    article_text = get_wikipedia_content(token, url)
    if not article_text:
        return

    logging.info("Content fetched. Summarizing...")
    summarized_text = summarize_text(token, article_text)

    logging.info("Text summarized. Generating audio...")
    audio_path = text_to_audio(token, summarized_text)
    if not audio_path:
        return

    # Create the video file
    logging.info("Audio generated. Creating video...")
    video_path = create_video(summarized_text, audio_path, token)

    # Delete the audio file immediately
    # Clean up audio immediately after using it
    if audio_path:
        os.remove(audio_path)

    # If video creation failed, log error and exit
    if not video_path:
        return

    # Upload video to S3 and clean up
    try:
        # Open the video file and pass the file object to the upload function
        with open(video_path, 'rb') as video_file:
            if upload_fileobj_to_s3(video_file, 'wikivid', f"{token}_output_video.mp4", acl='public-read'):
                video_url = f"https://wikivid.s3-ap-southeast-2.amazonaws.com/{token}_output_video.mp4"
                logging.info(f"Public URL of the video: {video_url}")
                finalize_request(token, video_url)
            else:
                raise Exception("Failed to upload video to S3.")
    except Exception as e:
        raise Exception(f"Failed to upload video to S3: {e}")
    finally:
        # Delete the video file immediately after uploading
        if os.path.exists(video_path):
            os.remove(video_path)


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
