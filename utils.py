import json
import re
from time import sleep

import requests

with open(".secrets.env") as f:
    secrets = json.loads(f.read())

import json


def get_abbrv(term: str, n, category=None):
    url = f"https://www.abbreviations.com/serp.php?st={term}&p=99999"
    resp: requests.Response = None
    while not resp or resp.status_code != 200:
        try:
            resp: requests.Response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                },
                timeout=20,
            )
        except requests.exceptions.Timeout:
            sleep(2)

    pattern = r'<p class="desc">(.*?)<\/p><p class="path"><a href="[^"]+">(.*?)<\/a>'

    definitions_prior = list()
    definitions = list()
    for full_form, cat in re.findall(pattern, resp.content.decode("utf-8")):
        if full_form not in definitions + definitions_prior:
            if not category or category == cat:
                definitions_prior.append(full_form)
            else:
                definitions.append(full_form)

    return (definitions_prior + definitions)[:n]


def get_abbrv2(term: str, n=5, category=None):
    def get(url):
        resp: requests.Response = None
        while not resp or resp.status_code != 200:
            try:
                resp: requests.Response = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    },
                    timeout=20,
                )
            except requests.exceptions.Timeout:
                sleep(2)

        json = resp.content

        # Find matches
        pattern = r'<td class="result-list__body__meaning">(?:<a href="[^"]+">)?([^<]+)(?:</a>)?</td>'

        return re.findall(pattern, json.decode("utf-8"))

    url1 = f"https://www.acronymfinder.com/Science-and-Medicine/{term}.html"
    url2 = f"https://www.acronymfinder.com/{term}.html"
    matches = get(url1)
    matches = matches + list(set(get(url2)).difference(set(matches)))
    return matches[:n]


def get_abbrv3(term: str, n=5, category=None):
    url = f"https://acronyms.thefreedictionary.com/{term}"
    resp: requests.Response = None
    while not resp or resp.status_code != 200:
        try:
            resp: requests.Response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                },
                timeout=20,
            )
        except requests.exceptions.Timeout:
            sleep(2)

    json = resp.content

    pattern = r"<td[^>]*>(.*?)</td>"

    # Find all matches using regex
    matches = list(
        set(
            map(
                lambda x: x.replace("<i>", "").replace("</i>", ""),
                filter(lambda x: term != x, re.findall(pattern, json.decode("utf-8"))),
            )
        )
    )

    # Extracted text

    return matches[:n]
