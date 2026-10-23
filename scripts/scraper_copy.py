import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TALK_URL= "https://www.churchofjesuschrist.org/study/general-conference/2026/10/28dunn?lang=eng"

def setup_session():
    """Create an HTTP session with a browser-like user agent."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session

def scrape_talk():
    html = ""
    try:
        html = requests.get(TALK_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).text
    except Exception:
        print("The talk could not be retrieved from the given URL.")

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    speaker_tag = soup.find("p", class_="author-name")
    calling_tag = soup.find("p", class_="author-role")
    text_tags = soup.select("div.body-block p")

    title = title_tag.text.strip()
    speaker = speaker_tag.text.strip()
    calling = calling_tag.text.strip()
    text = " ".join(p.text.strip() for p in text_tags)
    first_sentence = ""
    for word in text:
        first_sentence += word
        if "." in word:
            break

    print('The talk you found is ' + title + " by " + speaker + " who is " + calling + " and the first sentence is " + first_sentence)

scrape_talk()

