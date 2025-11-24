
# Allows type hints before they're defined
from __future__ import annotations

# A read and write Library for JSON Files
import json 

import logging

# Keeps paths as objects instead of raw strings
from pathlib import Path

# Type hints that help define tools
from typing import Any, Dict, List, Callable, Type, Optional, cast

# Hugging Face Transformers for FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Provides support for regular expressions
import re

# Summary of Goal: 
# Create a article sentimement scorer using C3AN pipeline 
# Breaks down the article into chunks that are scored for sentiments and aggregated. 



# ------------
# Logging Configuration
# ____________
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


# returns -1 for cpu usage for now because running into gpu issues
def select_device_for_hf() -> int:

    return -1

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        """Stub for Pydantic BaseModel if not installed."""
        model_fields: Dict[str, Any] = {}

        def __init_subclass__(cls, **kwargs: Any):
            super().__init_subclass__(**kwargs)

        def __init__(self, **data: Any):
            for k, v in data.items(): setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__.copy()

# Core classes
class Artifact(BaseModel):
    def to_json(self) -> Dict[str, Any]:
        return self.model_dump()

class Stage:
    # the init contains the parameters we will be using for the stages
    def __init__(
            self, 
            name: str,
            input_schema: Type[Artifact],
            output_schema: Type[Artifact],
            compute_fn: Callable[[Artifact], Artifact]
    ):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.compute_fn = compute_fn

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Validating input...")
        artifact_in = self.input_schema(**input_data)

        logger.info(f"[{self.name}] Computing...")
        artifact_out = self.compute_fn(artifact_in)

        logger.info(f"[{self.name}] Validating output...")
        artifact_valid = self.output_schema(**artifact_out.to_json())

        logger.info(f"[{self.name}] Completed successfully.")
        return artifact_valid.to_json()


class Pipeline:

    def __init__(self, name: str, stages: List[Stage]):
        self.name = name
        self.stages = stages

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Pipeline '{self.name}' starting...")
        for s in self.stages:
            data = s.run(data)
        logger.info(f"Pipeline '{self.name}' completed.")
        return data







# --------------
# Financial Article Pipeline: Clean Data -> Relevance Filter -> Sentiment -> Explanation
class ArticleArtifact(Artifact):
    title: Optional[str] = None
    text: str

class CleanArtifact(Artifact):
    paragraphs: List[str]
    original_count: int

class RelevantArtifact(Artifact):
    paragraphs: List[str]
    kept_indices: List[int]
    dropped_indices: List[int]

class ParagraphSentiment(Artifact):
    index: int
    text: str
    label: str
    scores: Dict[str, float]
    signed: float
    length: int

class SentimentArtifact(Artifact):
    per_paragraph: List[ParagraphSentiment]
    overall_signed: float
    hits: Dict[str, int]

class ArticleExplanationArtifact(Artifact):
    paragraph_explanations: List[str]
    overall_explanation: str










# ----- Stage 1 ------ Clean Data

_MIN_CHARS: int = 8

def normalize_and_chunk(text: str) -> List[str]:

    # resub.(pattern, replacement, string)
    t = re.sub(r'\r\n?', '\n', text)

    # Collapses all run on spaces into one space
    # [ \t] = a space
    # + = one or more in a row
    # strip removes all leading and trailing white spaces from string
    t = re.sub(r'[ \t]+', ' ', t).strip()

    # Checks for two different scenarios and splits string into a list of patterns
    #  \n\s*\n checks for a newline with any amount of white spaces 's*' and followed by another new line
    #  (?<=[.!?])\s{2,}, (?<= [.!?] is a lookbehind that ensures what before is followed up with . ! or ?
    parts = re.split(r'\n\s*\n|(?<=[.!?])\s{2,}', t)

    # loops over each chunk from the split and checks if the chunks are larger than the min character limit
    # Produces a cleaned list of paragraphs
    paras = [p.strip() for p in parts if p and len(p.strip()) >= _MIN_CHARS]

    # if paras has content its return or if its empty and has origin list return or if its empty return a empty list
    return paras or ([t] if t else [])


def clean_article(article: ArticleArtifact) -> CleanArtifact:
    logger.info("[Clean] Splitting into paragraphs")
    paragraphs =  normalize_and_chunk(article.text)

    # Returns the correct information defined in the CleanArtifact
    return CleanArtifact(paragraphs=paragraphs, original_count=len(paragraphs))

# Error identified here because in stages supposed to accept any artifact and return any artifact but this forces it to
# use only clean_article
CleanStage = Stage("Clean", ArticleArtifact, CleanArtifact, clean_article)











# ----- Stage 2 ----- Relevance Classifier

RELEVANCE_THRESHOLD: float = 0.55

ZS_Labels: List[str] = ["finance", "not finance"]

# The template can be changed, but will affect threshold
ZS_Template = "This text is about {}."


# Takes cleaned paragraphs from Clean Artifact, then uses bart model to score paragraph for finance label vs not finance keeps paragraphs
# that meet a certain threshold and filters the strings by kept and dropped indices
def get_zero_shot_pipeline():
    if not hasattr(get_zero_shot_pipeline, "zero_shot_classifier"):
        from transformers import pipeline
        import torch
        device_id = select_device_for_hf()
        get_zero_shot_pipeline.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            # Uses ready-to-use classifier
            model="facebook/bart-large-mnli",
            device=device_id
        )
    return get_zero_shot_pipeline.zero_shot_classifier



# If not cleaned paragraphs then it returns empty list, if it is a relevant paragraph its classified using bart and turned into dict
def relevance_filter(cleaned: CleanArtifact) -> RelevantArtifact:
    logger.info("[Relevance] Scoring paragraphs (zero-shot NLI)")
    # list of strings
    if not cleaned.paragraphs:
        # If not a paragraph then it returns an empty artifact
        return RelevantArtifact(paragraphs=[], kept_indices=[], dropped_indices=[])
    # Uses the classifier to score the cleaned paragraphs on if they are a finance article or not
    zero_shot_classifier = get_zero_shot_pipeline()
    results = zero_shot_classifier(
        cleaned.paragraphs,
        candidate_labels= ZS_Labels,
        hypothesis_template= ZS_Template,
        multi_label=False,
    )

    # Normalizes the results and makes sure its always a list
    if isinstance(results, dict):
        results = [results]



    kept: List[str] = []
    kept_indices: List[int] = []
    # for each paragraph and result it extracts the finance score
    for idx, (paragraph, result) in enumerate(zip(cleaned.paragraphs, results)):
        scores = dict(zip(result["labels"], result["scores"]))
        if scores.get("finance", 0.0) >= RELEVANCE_THRESHOLD:
            kept.append(paragraph)
            kept_indices.append(idx)

    # still keeps the paragraph if not relevant for potential useage
    if not kept:  # fail-open: keep all if nothing passes the threshold
        kept = cleaned.paragraphs[:]
        kept_indices = list(range(len(cleaned.paragraphs)))

    # edge case for setting i's that aren't kept_indices
    dropped_idx = [i for i in range(len(cleaned.paragraphs)) if i not in kept_indices]


    return RelevantArtifact(paragraphs=kept, kept_indices=kept_indices, dropped_indices=dropped_idx)

RelevanceStage = Stage("Relevance", CleanArtifact, RelevantArtifact, relevance_filter)













# -------- Stage 3 ------- Sentiment (Checks for sentiment Ex: positive, neutral, negative)

# Sentiment llm setup
def get_finbert_pipeline():
    if not hasattr(get_finbert_pipeline, "finbert_classifier"):
        from transformers import pipeline

        device_id = select_device_for_hf()  # -1 => CPU

        primary_id = "ProsusAI/finbert"
        fallback_id = "yiyanghkust/finbert-tone"  # stable, well-known FinBERT variant

        def build(model_id: str):
            return pipeline(
                task="sentiment-analysis",
                model=model_id,          # let HF resolve model + tokenizer
                tokenizer=model_id,
                device=device_id,
                top_k=None,              # return all scores (replacement for return_all_scores=True)
                truncation=True,
                max_length=256,
            )

        try:
            get_finbert_pipeline.finbert_classifier = build(primary_id)
            logger.info(f"[Sentiment] Loaded model: {primary_id}")
        except Exception as e:
            logger.warning(f"[Sentiment] Failed to load {primary_id}: {e}. Falling back to {fallback_id}.")
            get_finbert_pipeline.finbert_classifier = build(fallback_id)
            logger.info(f"[Sentiment] Loaded model: {fallback_id}")

        logger.info(f"[Sentiment] Device set to use {'cpu' if device_id == -1 else f'cuda:{device_id}'}")

    return get_finbert_pipeline.finbert_classifier

def scores_dict(all_scores: List[Dict[str, float]]) -> Dict[str, float]:

    # [{'label': 'positive', 'score': 0. 7}, ...] -> {'positive': 0.7, ...}
    dictionary = {e["label"].lower(): float(e["score"]) for e in all_scores}

    # tuple containing 3 strings for loop interates over and fills in 0.0 for empty label scores
    for k in ("positive", "neutral", "negative"):
        dictionary.setdefault(k, 0.0)
    return dictionary

# Finds the element with the highest postive sentiment score in the paragraphs
def label_of(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)

# Splits the paragraphs into words and returns the result accounting for 0
def words(paragraph: str) -> int:
    return max(1, len(paragraph.split()))


# Input: must contain a list of relevant paragraphs
# Output: is a structured SentimentArtifact that summarizes sentiment label and score, overall sentiment article and count of paragraphs with positive, negative or neutral
def finbert_sentiment(relevant: RelevantArtifact) -> SentimentArtifact:
    logger.info("[Sentiment] Scoring paragraphs with FinBert Model")

    if not relevant.paragraphs:
        return SentimentArtifact(
            per_paragraph=[],
            overall_signed =0.0,
            hits={"positive": 0, "neutral": 0, "negative": 0})

    finbert_pipeline = get_finbert_pipeline()

    # Gets the result and returns a list like ex: {"label": "positive", "score": 0.73
    all_results = finbert_pipeline(relevant.paragraphs)

    per_paragraph: List[ParagraphSentiment] = []
    pos = neu = neg = 0
    numerator: float = 0
    denominator: float = 0

    # pairs the paragraphs with its results
    # zip handles matching the paragraphs  and results
    # enumerate indexes the parings
    for idx, (paragraph, result) in enumerate(zip(relevant.paragraphs, all_results)):
        score_map = scores_dict(result)

        # chooses which label has the higest score
        label = label_of(score_map)

        # turns three-way probabilities into single number
        signed = score_map["positive"] - score_map["negative"]
        length = words(paragraph)

        # Tallies how many paragraphs fall into each sentiment class
        if label == "positive": pos += 1
        elif label == "negative": neg += 1
        else: neu += 1

        numerator += signed * length
        denominator += length

        # Creates a record and appends it the list of paragraphs
        per_paragraph.append(ParagraphSentiment(
            index = idx,
            text = paragraph,
            label = label,
            scores = score_map,
            signed = signed,
            length = length
        ))

    # Computes overall weight of sentiment
    overall = (numerator / denominator) if denominator else 0.0

    return SentimentArtifact(
        per_paragraph = per_paragraph,
        overall_signed = overall,
        hits = {"positive": pos, "neutral": neu, "negative": neg},
    )

SentimentStage = Stage("Sentiment", RelevantArtifact, SentimentArtifact, finbert_sentiment)











# ------- Stage 4 -------- Explanation

# This converts a float into percentage
def percentage_as_string(value: float) -> str:
    return f"{round(100 * value):d}%"

# Builds readable format for overall sentiment
def explain_article(artifact: Artifact) -> Artifact:
    sentiment_artifact = cast(SentimentArtifact, artifact)
    logger.info("[Explanation] Building explanations")

    # Creates a list of strings for explanations
    paragraph_explanations: List[str] = []
    # Creates the format [Paragraph 1} POSITIVE (postive = 73%, neutral = 18%, negative = 9%); signed_score += 0.41
    for paragraph_result in sentiment_artifact.per_paragraph:
        paragraph_scores = paragraph_result.scores
        paragraph_explanations.append(
            f"[Paragraph {paragraph_result.index + 1}] {paragraph_result.label.upper()} "
            f"(positive={percentage_as_string(paragraph_scores['positive'])}, "
            f"neutral={percentage_as_string(paragraph_scores['neutral'])}, "
            f"negative={percentage_as_string(paragraph_scores['negative'])}); "
            f"signed_score={paragraph_result.signed:+.2f}"
        )

    # Decides the overall label for each paragraph creating a deadzone
    if sentiment_artifact.overall_signed > 0.05:
        overall_label = "POSITIVE"
    elif sentiment_artifact.overall_signed < -0.05:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    # Formats results into format, Overall: POSITIVE (length-weighted signed score =+0.137). Count - Positive: 5, Neutral: 2, Negative: 1
    overall_explanation = (
        f"Overall: {overall_label} "
        f"(length-weighted signed score={sentiment_artifact.overall_signed:+.3f}). "
        f"Counts — Positive: {sentiment_artifact.hits['positive']}, "
        f"Neutral: {sentiment_artifact.hits['neutral']}, "
        f"Negative: {sentiment_artifact.hits['negative']}."
    )

    # Returns the object
    return ArticleExplanationArtifact(
        paragraph_explanations=paragraph_explanations,
        overall_explanation=overall_explanation,
    )

ExplanationStage = Stage(
"Explanation",
        SentimentArtifact,
        ArticleExplanationArtifact,
        explain_article)


# Runs the stages in order
article_pipeline = Pipeline(
    "ArticleSentiment",
    [CleanStage, RelevanceStage, SentimentStage, ExplanationStage]
)



if __name__ == "__main__":


    import os
    from pathlib import Path
    from typing import Optional, Tuple, List
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    # $env:KAGGLE_USERNAME = 'your name'
    # $env:KAGGLE_KEY = "your_key'
    api = KaggleApi()
    api.authenticate()

    # creates a ./dataset folder
    # keeps the data inside project so it's not user specific
    DATA_DIR = Path(__file__).parent / "dataset"
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    # Checks if the table already exists and if it doesn't then it unzips kaggle dataset
    print("Dataset URL: https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data")
    has_table = any(f.endswith((".csv", ".tsv", ".txt")) for f in os.listdir(DATA_DIR))
    if not has_table:
        logger.info("[Kaggle] Downloading dataset: aravsood7/sentiment-analysis-labelled-financial-news-data")
        api.dataset_download_files(
            "aravsood7/sentiment-analysis-labelled-financial-news-data",
            path=str(DATA_DIR),
            unzip=True,
        )
        logger.info("[Kaggle] Download complete.")
    else:
        logger.info("[Kaggle] Dataset already exists. Skipping download.")


    # After unzipping just lists everything under dataset/
    logger.info(f"[Kaggle] Listing files under: {DATA_DIR}")
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            logger.info(f" - {os.path.join(root, f)}")


    # Helper function that helps find candidate table files
    def find_candidate_table_files(base_dir: Path) -> List[Path]:
        # Recursively find CSV/TSV/TXT files under base_dir.
        exts = {".csv", ".tsv", ".txt"}
        files: List[Path] = []
        for p in base_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        return files

    # Tries to read table with pandas and if can't read then returns none
    def try_load_table(path: Path) -> Optional[pd.DataFrame]:
        for kwargs in (
            dict(),  # default pandas guesses
            dict(encoding="utf-8", on_bad_lines="skip"),
            dict(encoding="latin-1", on_bad_lines="skip"),
            dict(sep=None, engine="python", on_bad_lines="skip"),  # sniff delimiter
        ):
            try:
                df = pd.read_csv(path, **kwargs)
                if df is not None and df.shape[1] >= 1 and len(df) > 0:
                    return df
            except Exception:
                pass
        return None

    # Finds the column using naming convention if not found then takes the longest string
    def choose_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:

        lower_cols = {c.lower(): c for c in df.columns}

        # Choose text column
        text_col = None
        for key in ("text", "sentence", "headline", "content", "body", "article"):
            if key in lower_cols:
                text_col = lower_cols[key]
                break
        if text_col is None:
            # Fallback: longest average string length column
            best_col = None
            best_avg = -1.0
            for c in df.columns:
                try:
                    avg_len = df[c].astype(str).str.len().mean()
                    if avg_len > best_avg:
                        best_avg = avg_len
                        best_col = c
                except Exception:
                    continue
            text_col = best_col or df.columns[0]

        # Choose label column if available
        label_col = None
        for key in ("sentiment", "label", "target", "y"):
            if key in lower_cols:
                label_col = lower_cols[key]
                break

        return text_col, label_col


    # Finds and loads one table
    candidates = find_candidate_table_files(DATA_DIR)
    if not candidates:
        raise RuntimeError(
            "No CSV/TSV/TXT files found after unzip. Check the log for what was downloaded."
        )

    df = None
    chosen_file: Optional[Path] = None
    for p in candidates:
        df_try = try_load_table(p)
        if df_try is not None:
            df = df_try
            chosen_file = p
            break

    if df is None:
        raise RuntimeError("Could not load any table from the files found. Inspect the files listed above.")

    logger.info(f"[Loader] Using file: {chosen_file} (shape={df.shape})")

    # Detects which column is the text and label
    text_col, label_col = choose_text_and_label_columns(df)
    logger.info(
        f"[Loader] Selected text column: {text_col!r}"
        + (f", label column: {label_col!r}" if label_col else ", no label column found")
    )

    # Detects which column is text and label and runs pipeline on first row
    first_text = str(df.iloc[0][text_col])
    gold_label = str(df.iloc[0][label_col]) if label_col else None


    # Runs the pipeline and gets the result
    # Stage 1: Clean Stage -> paragraphs
    # Stage 2: Relevance Stage -> keeps finance-relevant paragraphs
    # Stage 3: Sentiment Stage -> FinBert scores + overall signed score
    # Stage 4: Explanation Stage -> human readable strings
    logger.info("----- Running pipeline on the FIRST dataset row -----")
    article_input = {"text": first_text}
    result = article_pipeline.run(article_input)

    # Prints out the results giving each paragraph's explanation and the overall explanation
    print("\n=== FIRST ARTICLE ===")
    if gold_label is not None:
        print(f"True Dataset Label: {gold_label}")
    print("\n--- Paragraph Explanations ---")
    for line in result.get("paragraph_explanations", []):
        print(line)
    print("\n--- Overall ---")
    print(result.get("overall_explanation", "<no overall explanation>"))







