import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm import embedder
from models import QueryAmbiguation


def get_embedding_likelihoods(df: pd.DataFrame, top_n: int, *, include_llm_suggestion: bool = False):
    for query_n in tqdm(range(len(df))):
        ambiguous_question: str = df.loc[query_n, "ambiguous_question"]
        ambiguities = QueryAmbiguation(**json.loads(df.loc[query_n, "possible_ambiguities"]))

        all_ambiguity_suggestions: list[list[str]] = [[amb.full_form] for amb in ambiguities.full_form_abbrv_map]
        # if there is a suggestion from APIs
        if isinstance(df.loc[query_n, f"top_{top_n}_full_form"], str):
            api_suggestions: list[list[str]] = json.loads(df.loc[query_n, f"top_{top_n}_full_form"])
            for i in range(len(api_suggestions)):
                all_ambiguity_suggestions[i] += api_suggestions[i]
            # add llm suggestion to df
        if include_llm_suggestion:
            llm_suggestions: list[str] = json.loads(df.loc[query_n, "llm_full_form_suggestions"])
            for i in range(len(llm_suggestions)):
                all_ambiguity_suggestions[i].append(llm_suggestions[i])

        ground_truth_full_form_probs = []
        llm_full_form_suggestion_probs = []
        most_likely_full_forms = []
        most_likely_full_form_probs = []
        most_likely_selection_type = []
        top_n_full_form_probs = []
        for suggestions, amb in zip(all_ambiguity_suggestions, ambiguities.full_form_abbrv_map):
            to_be_disambiguate_question = ambiguous_question.replace(
                amb.abbreviation, amb.abbreviation + " ({abbreviation})"
            )
            candidate_queries = [ambiguous_question] + [
                to_be_disambiguate_question.format(abbreviation=pos) for pos in suggestions
            ]

            embeddings = np.asarray(embedder.embed_documents(candidate_queries))

            similarities = (embeddings[:1, :] @ embeddings[1:, :].T).flatten().tolist()

            ground_truth_full_form_probs.append(similarities.pop(0))
            if include_llm_suggestion:
                llm_full_form_suggestion_probs.append(similarities[-1])

            if suggestions:
                arr = np.asarray(similarities)
                most_likely_index = arr.argmax()
                most_likely_full_forms.append(suggestions[most_likely_index])
                most_likely_full_form_probs.append(similarities[most_likely_index])
                most_likely_selection_type.append("API" if most_likely_index != (len(similarities) - 1) else "LLM")
                top_n_full_form_probs.append(similarities[:-1])

        df.loc[query_n, "TE_ground_truth_full_form_prob"] = json.dumps(ground_truth_full_form_probs)
        if include_llm_suggestion:
            df.loc[query_n, "TE_llm_full_form_suggestion_prob"] = json.dumps(llm_full_form_suggestion_probs)
        df.loc[query_n, "TE_most_likely_full_forms"] = json.dumps(most_likely_full_forms)
        df.loc[query_n, "TE_most_likely_full_form_probs"] = json.dumps(most_likely_full_form_probs)
        df.loc[query_n, "TE_most_likely_selection_types"] = json.dumps(most_likely_selection_type)
        df.loc[query_n, f"TE_top_{top_n}_full_form_probs"] = json.dumps(top_n_full_form_probs)
