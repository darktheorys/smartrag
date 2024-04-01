import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from models import QueryAmbiguation

device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased").to("cuda")

# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base").to("cuda")

# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
# model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base").to("cuda")

# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModelForMaskedLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to("cuda")
TOKENIZER = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
MLM = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT").to(device)

UNCASED = False


def get_mlm_likelihoods(df: pd.DataFrame, top_n: int, *, include_llm_suggestion: bool = False):
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
            to_be_masked_question = ambiguous_question.replace(amb.abbreviation, amb.abbreviation + " ({abbreviation})")
            suggestion_likelihoods = []
            for suggestion in suggestions:
                tokenized_suggestion = TOKENIZER(
                    suggestion.strip() if UNCASED else suggestion.casefold().strip(), return_tensors="pt"
                )
                # subtract 2 for [CLS] [SEP] tokens
                length = len(tokenized_suggestion.input_ids[0]) - 2

                # create masked suggestion
                masked_suggestion_state = [TOKENIZER.mask_token_id for _ in range(length)]

                logit_sum = 0.0
                for i in range(length):
                    # get next first token to calculate probability
                    current_token_id = tokenized_suggestion.input_ids[0][i + 1]

                    masked_disambiguated_question = to_be_masked_question.format(
                        abbreviation=TOKENIZER.decode(masked_suggestion_state)
                    )
                    tokenized_disambiguated_question = TOKENIZER(masked_disambiguated_question, return_tensors="pt").to(
                        device
                    )

                    mask_index = tokenized_disambiguated_question.input_ids[0].tolist().index(TOKENIZER.mask_token_id)
                    with torch.no_grad():
                        logits = torch.log_softmax(
                            MLM(**tokenized_disambiguated_question).logits[0, mask_index, :], dim=0
                        )
                    current_mask_logit = logits[current_token_id]

                    masked_suggestion_state[i] = current_token_id
                    logit_sum += current_mask_logit.item()
                suggestion_likelihoods.append(np.exp(logit_sum).item())

            ground_truth_full_form_probs.append(suggestion_likelihoods.pop(0))
            if include_llm_suggestion:
                llm_full_form_suggestion_probs.append(suggestion_likelihoods[-1])

            # remove gnd from suggestions
            suggestions.pop(0)
            if suggestions:
                arr = np.asarray(suggestion_likelihoods)
                most_likely_index = arr.argmax()
                most_likely_full_forms.append(suggestions[most_likely_index])
                most_likely_full_form_probs.append(suggestion_likelihoods[most_likely_index])
                most_likely_selection_type.append(
                    "API" if most_likely_index != (len(suggestion_likelihoods) - 1) else "LLM"
                )
                top_n_full_form_probs.append(suggestion_likelihoods[:-1])

        df.loc[query_n, "MLM_ground_truth_full_form_prob"] = json.dumps(ground_truth_full_form_probs)
        if include_llm_suggestion:
            df.loc[query_n, "MLM_llm_full_form_suggestion_prob"] = json.dumps(llm_full_form_suggestion_probs)
        df.loc[query_n, "MLM_most_likely_full_forms"] = json.dumps(most_likely_full_forms)
        df.loc[query_n, "MLM_most_likely_full_form_probs"] = json.dumps(most_likely_full_form_probs)
        df.loc[query_n, "MLM_most_likely_selection_types"] = json.dumps(most_likely_selection_type)
        df.loc[query_n, f"MLM_top_{top_n}_full_form_probs"] = json.dumps(top_n_full_form_probs)
