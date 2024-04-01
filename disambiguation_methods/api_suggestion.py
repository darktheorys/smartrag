import json
from time import sleep

import pandas as pd
from tqdm import tqdm

from models import QueryAmbiguation
from utils import get_abbrv, get_abbrv2, get_abbrv3, get_categories_with_regex


def get_abbreviation_suggestions(df: pd.DataFrame, top_n: int = 10) -> None:
    categories = []
    abbreviations = {}
    pbar = tqdm(total=len(df))
    for i in range(len(df)):
        url = f"https://www.abbreviations.com/category/{df.loc[i, "domain"]}"
        categories = get_categories_with_regex(url)

        ambiguities = QueryAmbiguation(**json.loads(df.loc[i, "possible_ambiguities"]))
        suggestions = []
        sources = []
        for amb in ambiguities.full_form_abbrv_map:
            if amb.abbreviation in abbreviations:
                suggestions.append(abbreviations[amb.abbreviation])
                pbar.update()
                continue

            popular_suggestions = get_abbrv(amb.abbreviation, top_n, categories=categories)
            suggestion_sources = ["ABBREVIATIONS"] * len(popular_suggestions)
            if len(popular_suggestions) < top_n:
                popular_suggestions_ = get_abbrv2(amb.abbreviation, top_n, categories=categories)
                popular_suggestions += popular_suggestions_
                suggestion_sources += ["ACRONYMFINDER"] * len(popular_suggestions_)
            if len(popular_suggestions) < top_n:
                popular_suggestions_ = get_abbrv3(amb.abbreviation, top_n, categories=categories)
                popular_suggestions += popular_suggestions_
                suggestion_sources += ["THEFREEDICTIONARY"] * len(popular_suggestions_)
            sleep(2)
            suggestions.append(list(set(popular_suggestions)))
            sources.append(suggestion_sources)
            abbreviations[amb.abbreviation] = suggestions[-1]
            pbar.update()
        df.loc[i, f"top_{top_n}_full_form"] = json.dumps(suggestions)
        df.loc[i, f"top_{top_n}_full_form_sources"] = json.dumps(sources)
