# Evaluation plan

The question set in `data/evaluation/questions.jsonl` is a starter evaluation set, not a benchmark. Use it to inspect retrieval and answer quality; do not claim a quality score until the questions have been run and results recorded.

## Manual review rubric

For each question, record:

1. **Retrieval hit:** Did at least one retrieved passage contain the expected source/evidence?
2. **Groundedness:** Are all factual claims supported by retrieved text?
3. **Citation correctness:** Do cited source labels point to the passages that support the claim?
4. **Abstention:** For unanswerable questions, did the app clearly say the corpus does not contain the answer?
5. **Usability:** Is the answer concise and understandable without losing important qualifications?

## Metrics to add

- Retrieval Hit@k: fraction of answerable questions whose expected source appears in the top k.
- MRR: average reciprocal rank of the first expected source.
- Answerable accuracy: human-rated groundedness and completeness.
- Abstention rate on unanswerable questions, alongside the rate of incorrect confident answers.
- Latency and API usage per question.

## Evaluation discipline

- Keep the evaluation questions separate from the documents used to build the index.
- Include questions about exclusions, limits, timing, and missing information.
- When changing chunking or retrieval settings, compare against the same question set.
- Record model names, settings, date, and known failures with each reported result.
- A small synthetic set is useful for iteration but cannot establish real-world reliability.
