import json
from time import sleep

import pandas as pd
from tqdm import tqdm

from models import QueryAmbiguation
from utils import get_abbrv, get_abbrv2, get_abbrv3, get_categories_with_regex


def get_abbreviation_suggestions(df: pd.DataFrame, top_n: int = 10, domain: str | None = None) -> None:
    categories = []
    if domain:
        url = f"https://www.abbreviations.com/category/{domain}"
        categories = get_categories_with_regex(url)

    abbreviations = {}
    pbar = tqdm(total=len(df))
    for i in range(len(df)):
        df.loc[i, f"top_{top_n}_full_form"] = ""
        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
        suggestions = []
        for amb in ambiguities.full_form_abbrv_map:
            if amb.abbreviation in abbreviations:
                suggestions.append(abbreviations[amb.abbreviation])
                pbar.update()
                continue

            popular_suggestions = get_abbrv(amb.abbreviation, top_n, categories=categories)
            if len(popular_suggestions) < top_n:
                popular_suggestions += get_abbrv2(amb.abbreviation, top_n, categories=categories)
            if len(popular_suggestions) < top_n:
                popular_suggestions += get_abbrv3(amb.abbreviation, top_n, categories=categories)
            sleep(2)
            suggestions.append(list(set(popular_suggestions)))
            abbreviations[amb.abbreviation] = suggestions[-1]
            pbar.update()
        df.loc[i, f"top_{top_n}_full_form"] = json.dumps(suggestions)
