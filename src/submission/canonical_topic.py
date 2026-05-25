import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError
from tavily import TavilyClient

from config import Config
from src.mongodbhandler import MongoDBHandler


logger = logging.getLogger(__name__)


class BuildCanonicalTopicError(Exception):
    def __init__(
        self,
        message: str,
        error_code: str,
        failed_stage: str,
        retryable: bool = False,
        exc_type: str = "ApplicationError",
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.failed_stage = failed_stage
        self.retryable = retryable
        self.exc_type = exc_type


class BuildCanonicalTopic:
    STAGES = [
        "validating_input",
        "generating_subtopics",
        "searching_sources",
        "extracting_claims",
        "clustering_claims",
        "verifying_claims",
        "writing_document",
        "embedding_claims",
        "persisting_document",
    ]

    def __init__(self, task=None):
        self.cfg = Config()
        self.task = task
        self.client = OpenAI()
        self.tavily_client = None
        self.usage: Dict[str, Any] = {
            "json_calls": 0,
            "embedding_calls": 0,
            "search_calls": 0,
            "json_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "embedding_usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "search_queries": [],
            "stage_durations": {},
        }

    def start(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        stage_started_at = time.time()

        try:
            self._progress("validating_input", 1)
            validated_submit_info = self._validate_submit_info(submit_info)
            effective_settings = self._resolve_settings(validated_submit_info)
            self._validate_runtime_dependencies()
            self.usage["stage_durations"]["validating_input"] = round(time.time() - stage_started_at, 3)

            dbname = self.cfg.canonical_topic_mongo_db_name
            collection_name = self.cfg.canonical_topic_collection_name
            mongoio = MongoDBHandler(dbname, collection_name)
            if not mongoio.is_online():
                raise BuildCanonicalTopicError(
                    "Internal MongoDB is offline",
                    error_code="MONGO_ERROR",
                    failed_stage="persisting_document",
                    retryable=True,
                )

            if mongoio.collection.find_one({"meta.doc_id": validated_submit_info["doc_id"]}) is not None:
                raise BuildCanonicalTopicError(
                    f'Document with doc_id={validated_submit_info["doc_id"]} already exists',
                    error_code="DUPLICATE_DOC_ID",
                    failed_stage="persisting_document",
                    retryable=False,
                    exc_type="ValueError",
                )

            topic = validated_submit_info["topic"]
            pipeline_started_at = time.time()

            subtopics = self._run_stage(
                "generating_subtopics",
                lambda: self.generate_subtopics(
                    topic=topic,
                    min_subtopics=effective_settings["min_subtopics"],
                    max_subtopics=effective_settings["max_subtopics"],
                    model=effective_settings["json_model"],
                ),
            )

            all_subtopic_outputs = []
            all_claims = []

            self._progress("searching_sources", 3, subtopic_count=len(subtopics), processed_subtopics=0)
            search_duration = 0.0
            extract_duration = 0.0
            for index, subtopic in enumerate(subtopics, start=1):
                search_started_at = time.time()
                sources = self.search_with_fallback(
                    topic=topic,
                    subtopic=subtopic,
                    min_sources=effective_settings["min_sources_per_subtopic"],
                    max_results=effective_settings["max_results_per_search"],
                    max_tier=effective_settings["search_max_tier"],
                )
                search_duration += time.time() - search_started_at

                self._progress(
                    "extracting_claims",
                    4,
                    subtopic_count=len(subtopics),
                    processed_subtopics=index,
                    current_subtopic=subtopic,
                    source_count=len(sources),
                )

                if not sources:
                    all_subtopic_outputs.append({
                        "subtopic": subtopic,
                        "warning": "No quality sources found.",
                        "claims": [],
                        "sources_used": [],
                    })
                    continue

                extract_started_at = time.time()
                extracted = self.extract_claims_from_sources(
                    topic=topic,
                    subtopic=subtopic,
                    sources=sources,
                    model=effective_settings["json_model"],
                )
                extract_duration += time.time() - extract_started_at

                extracted["sources_used"] = [
                    {
                        "source_id": source.get("source_id"),
                        "title": source.get("title"),
                        "url": source.get("url"),
                        "domain": source.get("domain"),
                        "source_quality_tier": source.get("source_quality_tier"),
                        "source_weight": source.get("source_weight"),
                    }
                    for source in sources
                ]

                all_subtopic_outputs.append(extracted)

                for claim in extracted.get("claims", []):
                    claim["subtopic"] = subtopic
                    all_claims.append(claim)

            self.usage["stage_durations"]["searching_sources"] = round(search_duration, 3)
            self.usage["stage_durations"]["extracting_claims"] = round(extract_duration, 3)

            clustered = self._run_stage(
                "clustering_claims",
                lambda: self.cluster_claims_strict(all_claims, effective_settings["json_model"]),
                claim_count=len(all_claims),
            )

            verified = self._run_stage(
                "verifying_claims",
                lambda: self.verify_clusters_weighted(clustered),
            )

            document = self._run_stage(
                "writing_document",
                lambda: self.write_canonical_document(
                    topic=topic,
                    subtopics=subtopics,
                    verified_claims=verified,
                    model=effective_settings["writer_model"],
                ),
            )

            embedded_verified = self._run_stage(
                "embedding_claims",
                lambda: self.embed_canonical_claims(
                    verified_claims=verified,
                    model=effective_settings["embedding_model"],
                ),
            )

            created_at = datetime.utcnow().isoformat()
            document["metadata"] = {
                "created_at": created_at,
                "topic": topic,
                "subtopics": subtopics,
                "pipeline_version": "v1_build_canonical_topic",
                "notes": [
                    "Sources filtered by quality tier before extraction.",
                    "Duplicate domains and near-identical content removed before trust scoring.",
                    "Trust labels use weighted source quality, not raw source count.",
                    "Low-trust claims are forbidden from canonical prose.",
                ],
            }

            canonical_topic_doc = {
                "meta": {
                    "doc_id": validated_submit_info["doc_id"],
                    "user_id": validated_submit_info["user_id"],
                    "type": "canonical_topic",
                    "topic": topic,
                    "status": "completed",
                    "pipeline_version": "v1_build_canonical_topic",
                    "creation_date_human": datetime.now().strftime("%B %d, %Y, %I:%M %p"),
                    "creation_time": time.time(),
                    "created_at": created_at,
                },
                "input": {
                    "submit_info": validated_submit_info,
                    "effective_settings": effective_settings,
                },
                "document": document,
                "intermediate": {
                    "subtopic_extractions": all_subtopic_outputs,
                    "clustered_claims": clustered,
                    "verified_claims": verified,
                    "embedded_verified_claims": embedded_verified,
                },
                "usage": {
                    **self.usage,
                    "total_runtime_seconds": round(time.time() - pipeline_started_at, 3),
                    "verified_claim_count": len(verified.get("verified_claims", [])),
                },
            }

            persist_started_at = time.time()
            self._progress(
                "persisting_document",
                9,
                verified_claim_count=len(verified.get("verified_claims", [])),
            )
            persisted_id = mongoio.write_document(canonical_topic_doc)
            self.usage["stage_durations"]["persisting_document"] = round(time.time() - persist_started_at, 3)

            logger.info(
                "build_canonical_topic succeeded: doc_id=%s topic=%s verified_claim_count=%s",
                validated_submit_info["doc_id"],
                topic,
                len(verified.get("verified_claims", [])),
            )
            return {
                "status": "SUCCESS",
                "message": "Canonical topic document has been saved",
                "doc_id": validated_submit_info["doc_id"],
                "canonical_topic_id": str(persisted_id),
                "topic": topic,
                "subtopic_count": len(subtopics),
                "verified_claim_count": len(verified.get("verified_claims", [])),
            }

        except BuildCanonicalTopicError as exc:
            logger.error(
                "build_canonical_topic failed: doc_id=%s stage=%s error_code=%s retryable=%s message=%s",
                submit_info.get("doc_id"),
                exc.failed_stage,
                exc.error_code,
                exc.retryable,
                exc.message,
            )
            return self._failure_dict(exc)
        except Exception as exc:
            logger.exception("build_canonical_topic unexpected failure: doc_id=%s", submit_info.get("doc_id"))
            wrapped = BuildCanonicalTopicError(
                str(exc),
                error_code="UNEXPECTED_ERROR",
                failed_stage="unexpected",
                retryable=False,
            )
            return self._failure_dict(wrapped)

    def _validate_submit_info(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["user_id", "doc_id", "topic"]
        missing_fields = [field for field in required_fields if not submit_info.get(field)]
        if missing_fields:
            raise BuildCanonicalTopicError(
                f"Missing required submit_info fields: {', '.join(missing_fields)}",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )

        return {
            key: value.strip() if isinstance(value, str) else value
            for key, value in submit_info.items()
        }

    def _resolve_settings(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        settings = {
            "min_subtopics": self._coerce_int(
                submit_info.get("min_subtopics"),
                self.cfg.canonical_topic_min_subtopics,
                "min_subtopics",
            ),
            "max_subtopics": self._coerce_int(
                submit_info.get("max_subtopics"),
                self.cfg.canonical_topic_max_subtopics,
                "max_subtopics",
            ),
            "min_sources_per_subtopic": self._coerce_int(
                submit_info.get("min_sources_per_subtopic"),
                self.cfg.canonical_topic_min_sources_per_subtopic,
                "min_sources_per_subtopic",
            ),
            "max_results_per_search": self._coerce_int(
                submit_info.get("max_results_per_search"),
                self.cfg.canonical_topic_max_results_per_search,
                "max_results_per_search",
            ),
            "search_max_tier": self._coerce_int(
                submit_info.get("search_max_tier"),
                self.cfg.canonical_topic_search_max_tier,
                "search_max_tier",
            ),
            "json_model": submit_info.get("json_model") or self.cfg.openai_canonical_topic_json_model,
            "writer_model": submit_info.get("writer_model") or self.cfg.openai_canonical_topic_writer_model,
            "embedding_model": submit_info.get("embedding_model") or self.cfg.openai_canonical_topic_embedding_model,
        }

        if settings["min_subtopics"] <= 0 or settings["max_subtopics"] <= 0:
            raise BuildCanonicalTopicError(
                "Subtopic settings must be positive integers",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )
        if settings["max_subtopics"] < settings["min_subtopics"]:
            raise BuildCanonicalTopicError(
                "max_subtopics must be greater than or equal to min_subtopics",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )
        if settings["min_sources_per_subtopic"] <= 0 or settings["max_results_per_search"] <= 0:
            raise BuildCanonicalTopicError(
                "Search settings must be positive integers",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )
        if settings["search_max_tier"] < 1 or settings["search_max_tier"] > 5:
            raise BuildCanonicalTopicError(
                "search_max_tier must be between 1 and 5",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )

        return settings

    def _validate_runtime_dependencies(self):
        if not self.cfg.tavily_api_key:
            raise BuildCanonicalTopicError(
                "TAVILY_API_KEY is not set",
                error_code="CONFIG_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="EnvironmentError",
            )

        self.tavily_client = TavilyClient(api_key=self.cfg.tavily_api_key)

    def _coerce_int(self, raw_value: Any, default_value: int, field_name: str) -> int:
        if raw_value in [None, ""]:
            return default_value

        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise BuildCanonicalTopicError(
                f"{field_name} must be an integer",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            ) from exc

    def _run_stage(self, stage_name: str, func, **progress_meta):
        stage_started_at = time.time()
        stage_index = self.STAGES.index(stage_name) + 1
        self._progress(stage_name, stage_index, **progress_meta)
        result = func()
        self.usage["stage_durations"][stage_name] = round(time.time() - stage_started_at, 3)
        return result

    def _progress(self, stage_name: str, stage_index: int, **meta):
        if self.task is not None:
            self.task.update_state(
                state="PROGRESS",
                meta={
                    "status": "PROGRESS",
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "stage_total": len(self.STAGES),
                    **meta,
                },
            )

    @staticmethod
    def get_domain(url: str) -> str:
        return urlparse(url).netloc.lower().replace("www.", "")

    def source_quality_tier(self, url: str) -> int:
        domain = self.get_domain(url)

        tier_1 = [
            "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "osf.io",
            "springer.com", "link.springer.com", "nature.com", "science.org", "sciencemag.org",
            "cell.com", "thelancet.com", "nejm.org", "bmj.com", "plos.org", "journals.plos.org",
            "wiley.com", "onlinelibrary.wiley.com", "tandfonline.com", "sagepub.com",
            "journals.sagepub.com", "cambridge.org", "oup.com", "academic.oup.com", "mdpi.com",
            "frontiersin.org", "elifesciences.org", "acm.org", "dl.acm.org", "ieee.org",
            "ieeexplore.ieee.org", "sciencedirect.com", "iopscience.iop.org", "aps.org",
            "journals.aps.org", "aip.org", "pubs.aip.org", "rsc.org", "pubs.rsc.org",
            "ams.org", "siam.org", "jmlr.org", "mlr.press", "proceedings.mlr.press",
            "openreview.net", "papers.nips.cc", "proceedings.neurips.cc", "aclanthology.org",
            "aclweb.org", "aaai.org", "ojs.aaai.org", "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov", "semanticscholar.org", "scholar.google.com",
        ]
        tier_2 = [
            ".edu", ".ac.uk", ".edu.au", ".edu.my", ".gov", ".gov.uk", ".gov.au",
            "europa.eu", "who.int", "oecd.org", "unesco.org", "un.org", "worldbank.org",
            "imf.org", "wto.org", "iaea.org", "ilo.org", "fao.org", "nih.gov", "cdc.gov",
            "nasa.gov", "noaa.gov", "nist.gov", "energy.gov", "nsf.gov", "cern.ch",
            "home.cern", "esa.int", "mpg.de", "max-planck.de", "cnrs.fr", "riken.jp",
            "wikipedia.org", "en.wikipedia.org", "scholarpedia.org", "plato.stanford.edu",
            "britannica.com", "encyclopedia.com", "wolframalpha.com", "mathworld.wolfram.com",
        ]
        tier_3 = [
            "nvidia.com", "developer.nvidia.com", "blogs.nvidia.com", "research.nvidia.com",
            "research.google", "ai.googleblog.com", "blog.google", "deepmind.google", "deepmind.com",
            "ai.meta.com", "research.facebook.com", "openai.com", "anthropic.com", "huggingface.co",
            "blog.huggingface.co", "mistral.ai", "cohere.com", "stability.ai", "x.ai",
            "microsoft.com/en-us/research", "research.microsoft.com", "research.ibm.com",
            "ibm.com/research", "apple.com/research", "machinelearning.apple.com", "amazon.science",
            "netflixtechblog.com", "engineering.fb.com", "tensorflow.org", "pytorch.org",
            "kaggle.com", "paperswithcode.com", "distill.pub", "github.com", "docs.python.org",
            "developer.mozilla.org", "developer.apple.com", "developer.android.com", "kubernetes.io",
            "cloud.google.com", "aws.amazon.com", "azure.microsoft.com",
        ]
        tier_5 = [
            "slideshare.net", "scribd.com", "linkedin.com", "medium.com", "towardsdatascience.com",
            "hackernoon.com", "dev.to", "prnewswire.com", "businesswire.com", "globenewswire.com",
            "substack.com", "quora.com", "reddit.com", "facebook.com", "twitter.com", "x.com",
            "youtube.com", "tiktok.com", "pinterest.com", "wordpress.com", "blogspot.com", "wix.com",
        ]

        if any(d in domain for d in tier_1):
            return 1
        if any(d in domain for d in tier_2):
            return 2
        if any(d in domain for d in tier_3):
            return 3
        if any(d in domain for d in tier_5):
            return 5
        return 4

    @staticmethod
    def source_weight(tier: int) -> float:
        return {
            1: 3.0,
            2: 2.0,
            3: 1.2,
            4: 0.7,
            5: 0.2,
        }.get(tier, 0.5)

    @staticmethod
    def content_hash(text: str, length: int = 2000) -> str:
        clean = re.sub(r"\s+", " ", text or "").strip().lower()
        clean = clean[:length]
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def dedupe_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_domains = set()
        seen_hashes = set()
        deduped = []

        for source in sources:
            url = source.get("url", "")
            raw_content = source.get("raw_content", "")

            domain = self.get_domain(url)
            content_hash = self.content_hash(raw_content)

            if domain in seen_domains or content_hash in seen_hashes:
                continue

            seen_domains.add(domain)
            seen_hashes.add(content_hash)
            deduped.append(source)

        return deduped

    def generate_subtopics(self, topic: str, min_subtopics: int, max_subtopics: int, model: str) -> List[str]:
        prompt = f"""
Break this topic into {min_subtopics} to {max_subtopics} important subtopics.

Topic:
{topic}

Rules:
- Subtopics must be useful for a canonical educational document.
- Avoid overlap.
- Keep subtopics short.
- Return JSON array only.
"""
        result = self.ask_openai_json(prompt, model=model, stage_name="generating_subtopics")
        if not isinstance(result, list):
            raise BuildCanonicalTopicError(
                "Subtopics must be a list",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="generating_subtopics",
                retryable=False,
            )

        subtopics = [str(item).strip() for item in result if str(item).strip()][:max_subtopics]
        if not subtopics:
            raise BuildCanonicalTopicError(
                "No subtopics were generated",
                error_code="DATA_PROCESSING_ERROR",
                failed_stage="generating_subtopics",
                retryable=False,
            )
        return subtopics

    def tavily_search_quality_sources(
        self,
        topic: str,
        subtopic: str,
        max_results: int,
        max_tier: int,
    ) -> List[Dict[str, Any]]:
        query = f"{topic} {subtopic} explanation research technical source"
        response = self._search_tavily(query=query, max_results=max_results)
        return self._extract_quality_sources(response, max_tier=max_tier)

    def search_with_fallback(
        self,
        topic: str,
        subtopic: str,
        min_sources: int,
        max_results: int,
        max_tier: int,
    ) -> List[Dict[str, Any]]:
        sources = self.tavily_search_quality_sources(
            topic=topic,
            subtopic=subtopic,
            max_results=max_results,
            max_tier=max_tier,
        )

        if len(sources) >= min_sources:
            return sources

        refined_queries = [
            f"{topic} {subtopic} site:arxiv.org OR site:ieee.org OR site:acm.org",
            f"{topic} {subtopic} research paper",
            f"{topic} {subtopic} university explanation",
        ]

        for query in refined_queries:
            response = self._search_tavily(query=query, max_results=max_results)
            sources.extend(self._extract_quality_sources(response, max_tier=max_tier))
            sources = self.dedupe_sources(sources)
            if len(sources) >= min_sources:
                break

        return sources

    def _search_tavily(self, query: str, max_results: int) -> Dict[str, Any]:
        for attempt in range(1, 4):
            try:
                self.usage["search_calls"] += 1
                self.usage["search_queries"].append(query)
                return self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=True,
                )
            except Exception as exc:
                if attempt == 3:
                    raise BuildCanonicalTopicError(
                        f"Tavily search failed: {exc}",
                        error_code="SEARCH_ERROR",
                        failed_stage="searching_sources",
                        retryable=True,
                    ) from exc
                time.sleep(1)

        raise BuildCanonicalTopicError(
            "Tavily search failed unexpectedly",
            error_code="SEARCH_ERROR",
            failed_stage="searching_sources",
            retryable=True,
        )

    def _extract_quality_sources(self, response: Dict[str, Any], max_tier: int) -> List[Dict[str, Any]]:
        sources = []

        for item in response.get("results", []):
            url = item.get("url")
            raw_content = item.get("raw_content") or item.get("content") or ""
            if not url or not raw_content:
                continue

            tier = self.source_quality_tier(url)
            if tier > max_tier:
                continue

            sources.append({
                "title": item.get("title"),
                "url": url,
                "domain": self.get_domain(url),
                "tavily_score": item.get("score"),
                "source_quality_tier": tier,
                "source_weight": self.source_weight(tier),
                "raw_content": raw_content[:self.cfg.canonical_topic_raw_content_limit],
            })

        return self.dedupe_sources(sources)

    def extract_claims_from_sources(
        self,
        topic: str,
        subtopic: str,
        sources: List[Dict[str, Any]],
        model: str,
    ) -> Dict[str, Any]:
        source_text = ""
        for index, source in enumerate(sources, start=1):
            source_id = f"S{index}"
            source["source_id"] = source_id
            source_text += f"""
SOURCE_ID: {source_id}
TITLE: {source["title"]}
URL: {source["url"]}
QUALITY_TIER: {source["source_quality_tier"]}
CONTENT:
{source["raw_content"]}
---
"""

        prompt = f"""
Topic:
{topic}

Subtopic:
{subtopic}

You are given raw source content.

Task:
Extract important factual claims.

Rules:
- Extract atomic claims only.
- One claim = one idea.
- Ignore ads, navigation, cookie text, author bios, and marketing fluff.
- Each claim must include a short verbatim quote from the source that supports it.
- Do not invent claims.
- Prefer technical, educational, and definitional information.
- Avoid vague claims like "X is revolutionary" unless the source provides concrete evidence.

Sources:
{source_text}

Return JSON only:

{{
  "subtopic": "{subtopic}",
  "claims": [
    {{
      "claim": "Atomic factual claim.",
      "source_id": "S1",
      "source_url": "https://example.com",
      "source_quality_tier": 1,
      "source_weight": 3.0,
      "source_quote": "Short exact quote supporting the claim."
    }}
  ]
}}
"""
        result = self.ask_openai_json(prompt, model=model, stage_name="extracting_claims")
        if not isinstance(result, dict):
            raise BuildCanonicalTopicError(
                "Claim extraction must return an object",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="extracting_claims",
                retryable=False,
            )
        return result

    def cluster_claims_strict(self, all_claims: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        if not all_claims:
            return {"claim_clusters": []}

        prompt = f"""
You are clustering extracted claims.

Task:
Group claims only if they express the SAME factual idea.

Important:
- Do NOT merge claims just because they are related.
- Preserve specificity.
- It is better to create more small clusters than fewer broad clusters.

Input:
{json.dumps(all_claims, indent=2)}

Return JSON only:

{{
  "claim_clusters": [
    {{
      "canonical_claim": "Most precise version of the shared claim.",
      "claims": [
        {{
          "claim": "Original claim",
          "source_url": "https://example.com",
          "source_quality_tier": 1,
          "source_weight": 3.0,
          "source_quote": "Quote"
        }}
      ]
    }}
  ]
}}
"""
        result = self.ask_openai_json(prompt, model=model, stage_name="clustering_claims")
        if not isinstance(result, dict):
            raise BuildCanonicalTopicError(
                "Claim clustering must return an object",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="clustering_claims",
                retryable=False,
            )
        return result

    def verify_clusters_weighted(self, claim_clusters: Dict[str, Any]) -> Dict[str, Any]:
        verified = []

        for cluster in claim_clusters.get("claim_clusters", []):
            claims = cluster.get("claims", [])
            unique_domains = set()
            unique_urls = set()
            weighted_score = 0.0
            best_tier = 5
            quotes = []

            for claim in claims:
                url = claim.get("source_url", "")
                domain = self.get_domain(url)

                if domain in unique_domains:
                    continue

                unique_domains.add(domain)
                unique_urls.add(url)

                tier = int(claim.get("source_quality_tier", 5))
                weight = float(claim.get("source_weight", self.source_weight(tier)))
                weighted_score += weight
                best_tier = min(best_tier, tier)

                quotes.append({
                    "source_url": url,
                    "source_quality_tier": tier,
                    "source_quote": claim.get("source_quote", ""),
                })

            if weighted_score >= 5.0 and best_tier <= 2:
                trust_label = "high"
            elif weighted_score >= 3.0:
                trust_label = "medium"
            elif weighted_score >= 1.0:
                trust_label = "low"
            else:
                trust_label = "unverified"

            verified.append({
                "canonical_claim": cluster.get("canonical_claim"),
                "trust_label": trust_label,
                "weighted_support_score": round(weighted_score, 2),
                "unique_source_count": len(unique_urls),
                "supporting_source_urls": list(unique_urls),
                "supporting_quotes": quotes,
            })

        return {"verified_claims": verified}

    def write_canonical_document(
        self,
        topic: str,
        subtopics: List[str],
        verified_claims: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        prompt = f"""
You are writing a canonical educational document.

Topic:
{topic}

Subtopics:
{json.dumps(subtopics, indent=2)}

Verified claims:
{json.dumps(verified_claims, indent=2)}

Rules:
- Use only high and medium trust claims in canonical_text.
- Do NOT state low-trust claims as fact in prose.
- Low-trust claims may appear only in a separate "low_trust_notes" field.
- Do not include unverified claims.
- Every key point must include trust_label and sources.
- Keep the text clean, neutral, and educational.
- Avoid hype and unsupported future predictions.
- If evidence is weak, say the evidence is weak.

Return JSON only:

{{
  "topic": "{topic}",
  "canonical_summary": "Brief summary using only medium/high trust claims.",
  "sections": [
    {{
      "subtopic": "Subtopic name",
      "canonical_text": "Explanation using only high/medium trust claims.",
      "key_points": [
        {{
          "point": "Supported point.",
          "trust_label": "high",
          "sources": ["https://source.com"]
        }}
      ],
      "low_trust_notes": [
        {{
          "point": "Some sources suggest...",
          "sources": ["https://source.com"]
        }}
      ]
    }}
  ],
  "source_index": [
    {{
      "url": "https://source.com",
      "claims_supported": ["Claim here"]
    }}
  ]
}}
"""
        result = self.ask_openai_json(prompt, model=model, temperature=0.1, stage_name="writing_document")
        if not isinstance(result, dict):
            raise BuildCanonicalTopicError(
                "Canonical document writer must return an object",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="writing_document",
                retryable=False,
            )
        return result

    def embed_canonical_claims(self, verified_claims: Dict[str, Any], model: str) -> Dict[str, Any]:
        embedded_claims = []
        for item in verified_claims.get("verified_claims", []):
            claim_text = item.get("canonical_claim", "")
            if not claim_text:
                continue

            for attempt in range(1, 4):
                try:
                    response = self.client.embeddings.create(model=model, input=claim_text)
                    self.usage["embedding_calls"] += 1
                    self._merge_embedding_usage(getattr(response, "usage", None))
                    embedded_claims.append({
                        **item,
                        "embedding_model": model,
                        "embedding": response.data[0].embedding,
                    })
                    break
                except BadRequestError as exc:
                    raise BuildCanonicalTopicError(
                        self._format_openai_error(exc),
                        error_code="EMBEDDING_ERROR",
                        failed_stage="embedding_claims",
                        retryable=False,
                    ) from exc
                except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                    if attempt == 3:
                        raise BuildCanonicalTopicError(
                            self._format_openai_error(exc),
                            error_code="EMBEDDING_ERROR",
                            failed_stage="embedding_claims",
                            retryable=True,
                        ) from exc
                    time.sleep(1)
                except APIError as exc:
                    raise BuildCanonicalTopicError(
                        self._format_openai_error(exc),
                        error_code="EMBEDDING_ERROR",
                        failed_stage="embedding_claims",
                        retryable=False,
                    ) from exc

        return {"verified_claims": embedded_claims}

    def ask_openai_json(
        self,
        prompt: str,
        model: str,
        stage_name: str,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> Any:
        last_error = None
        last_content = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful research assistant. "
                                "Return valid JSON only. No markdown. "
                                "Do not include explanations outside JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                self.usage["json_calls"] += 1
                self._merge_chat_usage(getattr(response, "usage", None))

                content = response.choices[0].message.content
                last_content = content
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    last_error = exc
                    match = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
            except BadRequestError as exc:
                raise BuildCanonicalTopicError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_error = exc
                if attempt == max_retries:
                    raise BuildCanonicalTopicError(
                        self._format_openai_error(exc),
                        error_code="OPENAI_API_ERROR",
                        failed_stage=stage_name,
                        retryable=True,
                    ) from exc
                time.sleep(1)
            except APIError as exc:
                raise BuildCanonicalTopicError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc

        raise BuildCanonicalTopicError(
            (
                f"OpenAI failed to return valid JSON after {max_retries} attempts. "
                f"Last error: {last_error}. Last content: {last_content}"
            ),
            error_code="OPENAI_JSON_ERROR",
            failed_stage=stage_name,
            retryable=False,
        )

    @staticmethod
    def _format_openai_error(exc: Exception) -> str:
        body = getattr(exc, "body", None)
        request_id = getattr(exc, "request_id", None)
        detail = body if body is not None else str(exc)
        if request_id:
            return f"OpenAI request failed: {detail} (request_id={request_id})"
        return f"OpenAI request failed: {detail}"

    def _merge_chat_usage(self, usage):
        usage_dict = self._usage_to_dict(usage)
        self.usage["json_usage"]["prompt_tokens"] += usage_dict.get("prompt_tokens", 0)
        self.usage["json_usage"]["completion_tokens"] += usage_dict.get("completion_tokens", 0)
        self.usage["json_usage"]["total_tokens"] += usage_dict.get("total_tokens", 0)

    def _merge_embedding_usage(self, usage):
        usage_dict = self._usage_to_dict(usage)
        self.usage["embedding_usage"]["prompt_tokens"] += usage_dict.get("prompt_tokens", 0)
        self.usage["embedding_usage"]["total_tokens"] += usage_dict.get("total_tokens", 0)

    @staticmethod
    def _usage_to_dict(usage) -> Dict[str, int]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    @staticmethod
    def _failure_dict(exc: BuildCanonicalTopicError) -> Dict[str, Any]:
        return {
            "status": "FAILED",
            "message": exc.message,
            "error_code": exc.error_code,
            "failed_stage": exc.failed_stage,
            "retryable": exc.retryable,
            "exc_type": exc.exc_type,
            "exc_message": exc.message,
        }
