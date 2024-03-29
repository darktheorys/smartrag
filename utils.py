import json
import re

import requests

with open(".secrets.env") as f:
    secrets = json.loads(f.read())


def get_abbrv(term: str, n, categories=[]):
    url = f"https://www.stands4.com/services/v2/abbr.php?uid={secrets.get('abbrv_userid')}&tokenid={secrets.get('abbrv_token')}&term={term}&format=json"
    resp: requests.Response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        },
    )
    if resp.status_code != 200:
        print(resp.content)
        raise RuntimeError()

    json = resp.json()
    definitions_prior = list()
    definitions = list()
    if "result" in json:
        if isinstance(json["result"], dict):
            json["result"] = [json["result"]]
        for res in json["result"]:
            if res["definition"] not in definitions + definitions_prior:
                if not categories or res["category"] in categories:
                    definitions_prior.append(res["definition"])
                else:
                    definitions.append(res["definition"])

    return (definitions_prior + definitions)[:n]


def get_abbrv2(term: str, n=5, categories=None):
    def get(url):
        resp: requests.Response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            },
        )
        if resp.status_code != 200:
            print(resp.content)
            raise RuntimeError()

        json = resp.content

        # Find matches
        pattern = r'<td class="result-list__body__meaning">(?:<a href="[^"]+">)?([^<]+)(?:</a>)?</td>'

        return re.findall(pattern, json.decode("utf-8"))

    url1 = f"https://www.acronymfinder.com/Science-and-Medicine/{term}.html"
    url2 = f"https://www.acronymfinder.com/{term}.html"
    matches = get(url1)
    matches = matches + list(set(get(url2)).difference(set(matches)))
    return matches[:n]


def get_abbrv3(term: str, n=5, categories=None):
    url = f"https://acronyms.thefreedictionary.com/{term}"
    resp: requests.Response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        },
    )
    if resp.status_code != 200:
        print(resp.content)
        raise RuntimeError()

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


def get_categories_with_regex(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        },
    )
    pattern = r'href="/acronyms/([^"]+)"'
    links = re.findall(pattern, response.text)
    return links
