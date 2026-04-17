"""NLP agent: NER, classification, summarization, language, key phrases."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from textwrap import dedent

from langdetect import DetectorFactory, LangDetectException, detect

from backend.core.config import get_settings
from backend.core.llm_client import generate
from backend.core.logging import get_logger
from backend.core.models import DocumentType, Entity, NLPAnalysis
from backend.pipelines.ingestion_pipeline import _classify_heuristic  # type: ignore[attr-defined]

DetectorFactory.seed = 42

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_spacy():
    """Load a small, multilingual-friendly spaCy pipeline.

    Falls back to the blank English pipeline if the model is not installed,
    which keeps the app functional while tests can still run.
    """
    import spacy

    for model_name in ("xx_ent_wiki_sm", "en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    log.warning("No spaCy model installed, falling back to blank('en')")
    return spacy.blank("en")


_SUMMARY_PROMPT = dedent(
    """
    Summarise the following document in 4-6 sentences. Keep it factual,
    preserve numbers, and do not invent information.

    Document:
    {text}

    Summary:
    """
).strip()


_STOPWORDS = set(
    """
    the a an and or but if to of in on at for by with from as is are was were
    be been being this that these those it its they them their our your you
    we i he she his her not no do does did have has had will would can could
    should which who whom whose what when where why how
    и в на с по за к о от у из не что это как так но а или да же бы
    """.split()
)


def _keyphrases(text: str, top_n: int = 8) -> list[str]:
    """Very simple TF-based key-phrase extractor — language agnostic."""
    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{2,}", text)
    ]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    if not tokens:
        return []
    counts = Counter(tokens)
    # Extract bigrams too
    bigrams = [
        f"{a} {b}"
        for a, b in zip(tokens, tokens[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS
    ]
    counts.update(Counter(bigrams))
    return [w for w, _ in counts.most_common(top_n)]


class NLPAgent:
    """spaCy + local llama.cpp NLP feature extractor."""

    def __init__(self) -> None:
        self.settings = get_settings()

    # ---- individual tasks ---- #

    def entities(self, text: str) -> list[Entity]:
        nlp = _load_spacy()
        if "ner" not in nlp.pipe_names:
            return []
        doc = nlp(text[:20000])
        wanted = {"PERSON", "PER", "ORG", "GPE", "LOC", "DATE", "MONEY", "TIME", "PRODUCT"}
        seen: set[tuple[str, str]] = set()
        out: list[Entity] = []
        for ent in doc.ents:
            if ent.label_ not in wanted:
                continue
            key = (ent.text.strip(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Entity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
        return out

    def classify(self, text: str) -> DocumentType:
        return _classify_heuristic(text)

    def detect_language(self, text: str) -> str:
        try:
            return detect(text[:2000])
        except LangDetectException:
            return "unknown"

    def summarize(self, text: str) -> str:
        snippet = text.strip()[:6000]
        if not snippet:
            return ""
        try:
            return generate(
                _SUMMARY_PROMPT.format(text=snippet),
                max_tokens=300,
                temperature=0.2,
            )
        except Exception as exc:
            log.error(f"Summarization failed: {exc}")
            return snippet[:400] + ("..." if len(snippet) > 400 else "")

    def keyphrases(self, text: str, top_n: int = 8) -> list[str]:
        return _keyphrases(text, top_n=top_n)

    # ---- full analyse ---- #

    def analyze(self, text: str, tasks: list[str]) -> NLPAnalysis:
        """Run the requested NLP analysis tasks on ``text``.

        Args:
            text: Raw text to analyse. Long inputs are truncated
                internally for the LLM summariser only.
            tasks: Subset of
                ``["entities", "classify", "summarize", "language", "keyphrases"]``.
                Unknown task names are silently ignored.

        Returns:
            An :class:`NLPAnalysis` with only the requested fields
            populated; others keep their schema defaults.
        """
        result = NLPAnalysis()
        if "entities" in tasks:
            result.entities = self.entities(text)
        if "classify" in tasks:
            result.classification = self.classify(text)
        if "language" in tasks:
            result.language = self.detect_language(text)
        if "summarize" in tasks:
            result.summary = self.summarize(text)
        if "keyphrases" in tasks:
            result.keyphrases = self.keyphrases(text)
        return result
