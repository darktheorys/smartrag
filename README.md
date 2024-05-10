# smartrag

Smartrag is a project aimed at smarter and context-aware retrieval augmented generation with LLMs.

## File Structure

- data/
  - This directory contains various data used in the project, including AbbrvQA, MedQuAD, figures, and others like MedQuAD, StrategyQA, BoolQ, FinQABench, ....
- disambiguation_methods/
  - This directory contains Python scripts for various disambiguation methods, including ambiguity extraction, API suggestion, domain extraction, dtype extraction, embedding similarity, generating abbreviations, and intent extraction.
- llm.py
  - This Python script contains the main logic for the language model.
- models.py
  - This Python script contains the data models used in the project.
- notebooks/
  - This directory contains Jupyter notebooks, which are used for interactive data analysis and visualization.
- prompt.py
  - This Python script contains the logic for generating RAG strategy prompt.
- qa_ambiguous_with_top10_mlm.csv and qa_ambiguous_with_top10.csv - These CSV files contain question-answer pairs with ambiguity. For testring MLM and TE with different top-k and glossaries.
- rag/
  - This directory contains scripts related to the retrieval-augmented generation model.
- strategy_example.json
  - This JSON file contains an example of a strategy obtained for RAG pipeline.
- utils.py
  - This Python script contains utility functions used across the project.

Notebooks
The project includes several Jupyter notebooks for data analysis and visualization. These notebooks include code for reading CSV files, generating queries, and processing disambiguation methods.

License
The project is licensed under the Apache License 2.0.
