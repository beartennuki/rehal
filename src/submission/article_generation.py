import json
import logging
import math
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from bson import ObjectId
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError

from config import Config
from src.mongodbhandler import MongoDBHandler
from src.submission.article_quality import ArticleQualityChecker


logger = logging.getLogger(__name__)


class ArticleGenerationError(Exception):
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


class GenerateArticle:
    STAGES = [
        "validating_input",
        "loading_canonical_document",
        "extracting_claim_bank",
        "generating_outline",
        "retrieving_claims",
        "filtering_claims",
        "writing_sections",
        "writing_intro_conclusion",
        "rendering_article",
        "quality_check",
        "repairing_article",
        "persisting_document",
    ]

    CLAIM_TAG_RE = re.compile(r"\[(C\d{4})\]")
    SOURCE_TAG_RE = re.compile(r"\[\^(S\d{3})\]")
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, task=None):
        self.cfg = Config()
        self.task = task
        self.client = OpenAI()
        self.usage: Dict[str, Any] = {
            "json_calls": 0,
            "text_calls": 0,
            "embedding_calls": 0,
            "retries": 0,
            "json_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "text_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "embedding_usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "stage_durations": {},
        }
        self.claim_bank: List[Dict[str, Any]] = []
        self.claim_id_to_claim: Dict[str, Dict[str, Any]] = {}
        self.sid_to_url: Dict[str, str] = {}
        self.quality_checker: ArticleQualityChecker = ArticleQualityChecker({}, {}, [])

    def start(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.time()
        article_id = uuid4().hex
        validated_submit_info = None
        settings = None
        canonical_doc = None
        source_context = None
        outline = None
        raw_retrieval = None
        retrieval = None
        last_body_sections: List[Dict[str, Any]] = []
        last_intro_conclusion = {"introduction": "", "conclusion": ""}
        last_rendered = {
            "article_with_claim_ids": "",
            "article_with_sources": "",
            "used_source_ids": [],
        }
        last_quality = None
        attempt_history: List[Dict[str, Any]] = []
        embedding_doc_id = None

        try:
            validated_submit_info = self._run_stage(
                "validating_input",
                lambda: self._validate_submit_info(submit_info),
            )
            settings = self._resolve_settings(validated_submit_info)

            canonical_mongo = MongoDBHandler(
                self.cfg.canonical_topic_mongo_db_name,
                self.cfg.canonical_topic_collection_name,
            )
            article_mongo = MongoDBHandler(
                self.cfg.generated_article_mongo_db_name,
                self.cfg.generated_article_collection_name,
            )
            embedding_mongo = MongoDBHandler(
                self.cfg.generated_article_mongo_db_name,
                self.cfg.article_embedding_collection_name,
            )

            if not canonical_mongo.is_online() or not article_mongo.is_online() or not embedding_mongo.is_online():
                raise ArticleGenerationError(
                    "Internal MongoDB is offline",
                    error_code="MONGO_ERROR",
                    failed_stage="loading_canonical_document",
                    retryable=True,
                )

            canonical_doc = self._run_stage(
                "loading_canonical_document",
                lambda: self._load_canonical_document(validated_submit_info["canonical_doc_id"], canonical_mongo),
            )

            source_context = self._run_stage(
                "extracting_claim_bank",
                lambda: self._build_source_context(canonical_doc, settings["embedding_model"]),
            )

            try:
                embedding_doc_id = str(self._persist_embedding_document(
                    self._build_embedding_document(article_id, validated_submit_info, source_context, settings),
                    embedding_mongo,
                ))
            except ArticleGenerationError as exc:
                logger.warning(
                    "generate_article failed to persist embedding document: article_id=%s error=%s",
                    article_id,
                    exc.message,
                )

            outline = self._run_stage(
                "generating_outline",
                lambda: self.generate_outline(
                    topic=source_context["topic"],
                    canonical_summary=source_context["canonical_summary"],
                    subtopics=source_context["subtopics"],
                    difficulty_level=validated_submit_info["difficulty_level"],
                    personal_remarks=validated_submit_info["personal_remarks"],
                    model=settings["outline_model"],
                ),
                subtopic_count=len(source_context["subtopics"]),
            )

            raw_retrieval = self._run_stage(
                "retrieving_claims",
                lambda: self.retrieve_claims_for_outline(
                    outline=outline,
                    claim_bank=self.claim_bank,
                    difficulty_level=validated_submit_info["difficulty_level"],
                    personal_remarks=validated_submit_info["personal_remarks"],
                    embedding_model=settings["embedding_model"],
                    top_k=settings["top_k_per_point"],
                ),
            )

            retrieval = self._run_stage(
                "filtering_claims",
                lambda: self.merge_and_filter_retrieval(
                    raw_retrieval,
                    settings["near_duplicate_threshold"],
                    settings["max_claims_per_section"],
                ),
            )

            for attempt_number in range(1, self.MAX_REPAIR_ATTEMPTS + 1):
                attempt_type = "initial" if attempt_number == 1 else "repair"

                if attempt_number == 1:
                    body_sections = self._run_stage(
                        "writing_sections",
                        lambda: self.write_body_sections(
                            retrieval=retrieval,
                            outline=outline,
                            difficulty_level=validated_submit_info["difficulty_level"],
                            personal_remarks=validated_submit_info["personal_remarks"],
                            writer_model=settings["writer_model"],
                            target_word_count=settings["target_word_count"],
                        ),
                        attempt_number=attempt_number,
                        max_attempts=self.MAX_REPAIR_ATTEMPTS,
                    )

                    intro_conclusion = self._run_stage(
                        "writing_intro_conclusion",
                        lambda: self.write_intro_and_conclusion(
                            topic=source_context["topic"],
                            canonical_summary=source_context["canonical_summary"],
                            body_sections=body_sections,
                            difficulty_level=validated_submit_info["difficulty_level"],
                            personal_remarks=validated_submit_info["personal_remarks"],
                            model=settings["intro_model"],
                        ),
                        attempt_number=attempt_number,
                        max_attempts=self.MAX_REPAIR_ATTEMPTS,
                    )
                else:
                    repaired = self._run_stage(
                        "repairing_article",
                        lambda: self.repair_article_attempt(
                            retrieval=retrieval,
                            outline=outline,
                            topic=source_context["topic"],
                            canonical_summary=source_context["canonical_summary"],
                            body_sections=last_body_sections,
                            intro_conclusion=last_intro_conclusion,
                            quality=last_quality,
                            difficulty_level=validated_submit_info["difficulty_level"],
                            personal_remarks=validated_submit_info["personal_remarks"],
                            writer_model=settings["writer_model"],
                            intro_model=settings["intro_model"],
                            target_word_count=settings["target_word_count"],
                            attempt_number=attempt_number,
                        ),
                        attempt_number=attempt_number,
                        max_attempts=self.MAX_REPAIR_ATTEMPTS,
                    )
                    body_sections = repaired["body_sections"]
                    intro_conclusion = repaired["intro_conclusion"]
                    repair_instructions = repaired["repair_instructions"]
                rendered = self._run_stage(
                    "rendering_article",
                    lambda: self.render_article(
                        topic=source_context["topic"],
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                    ),
                    attempt_number=attempt_number,
                    max_attempts=self.MAX_REPAIR_ATTEMPTS,
                )

                quality = self._run_stage(
                    "quality_check",
                    lambda: self.run_quality_checks_with_openai(
                        retrieval=retrieval,
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                        rendered=rendered,
                        qa_model=settings["qa_model"],
                    ),
                    attempt_number=attempt_number,
                    max_attempts=self.MAX_REPAIR_ATTEMPTS,
                )

                if "untagged_factual_sentences" in quality["failures"]:
                    body_sections, intro_conclusion = self.repair_untagged_paragraphs_with_openai(
                        retrieval=retrieval,
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                        findings=quality["findings"],
                        writer_model=settings["writer_model"],
                    )
                    rendered = self.render_article(
                        topic=source_context["topic"],
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                    )
                    quality = self.run_quality_checks_with_openai(
                        retrieval=retrieval,
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                        rendered=rendered,
                        qa_model=settings["qa_model"],
                    )

                last_body_sections = body_sections
                last_intro_conclusion = intro_conclusion
                last_rendered = rendered
                last_quality = quality
                attempt_history.append(
                    self.build_attempt_record(
                        attempt_number=attempt_number,
                        attempt_type=attempt_type,
                        body_sections=body_sections,
                        intro_conclusion=intro_conclusion,
                        rendered=rendered,
                        quality=quality,
                        repair_instructions=None if attempt_number == 1 else repair_instructions,
                    )
                )

                logger.info(
                    "generate_article attempt=%s/%s passed=%s failures=%s",
                    attempt_number,
                    self.MAX_REPAIR_ATTEMPTS,
                    quality["passed"],
                    quality["failures"],
                )

                if quality["passed"]:
                    break
                if not quality["repairable"]:
                    break

            final_status = "completed" if last_quality and last_quality["passed"] else "failed"
            article_doc = self._build_article_document(
                article_id=article_id,
                submit_info=validated_submit_info,
                settings=settings,
                canonical_doc=canonical_doc,
                source_context=source_context,
                outline=outline,
                raw_retrieval=raw_retrieval,
                retrieval=retrieval,
                body_sections=last_body_sections,
                intro_conclusion=last_intro_conclusion,
                rendered=last_rendered,
                quality=last_quality or self.quality_checker.build_exception_quality("No quality results were produced"),
                attempt_history=attempt_history,
                status=final_status,
                final_error_code=None if final_status == "completed" else "QUALITY_CHECK_FAILED",
                final_failed_stage=None if final_status == "completed" else "quality_check",
                total_runtime_seconds=round(time.time() - started_at, 3),
                embedding_doc_id=embedding_doc_id,
            )

            persist_started_at = time.time()
            self._progress(
                "persisting_document",
                self.STAGES.index("persisting_document") + 1,
                citation_count=(last_quality or {}).get("coverage_report", {}).get("total_cited_claims", 0),
                attempt_count=len(attempt_history),
            )
            persisted_id = self._persist_article_document(article_doc, article_mongo)
            self.usage["stage_durations"]["persisting_document"] = round(time.time() - persist_started_at, 3)

            if final_status == "completed":
                logger.info(
                    "generate_article succeeded: article_id=%s canonical_doc_id=%s citations=%s attempts=%s",
                    article_id,
                    validated_submit_info["canonical_doc_id"],
                    last_quality["coverage_report"]["total_cited_claims"],
                    len(attempt_history),
                )
                return {
                    "status": "SUCCESS",
                    "message": "Article document has been saved",
                    "article_id": article_id,
                    "generated_article_id": str(persisted_id),
                    "canonical_doc_id": validated_submit_info["canonical_doc_id"],
                    "difficulty_level": validated_submit_info["difficulty_level"],
                    "subtopic_count": len(source_context["subtopics"]),
                    "section_count": len(last_body_sections),
                    "attempt_count": len(attempt_history),
                    "max_attempts": self.MAX_REPAIR_ATTEMPTS,
                    "quality_summary": self.quality_checker.build_quality_summary(last_quality),
                }

            logger.error(
                "generate_article exhausted repairs: article_id=%s canonical_doc_id=%s attempts=%s failures=%s",
                article_id,
                validated_submit_info["canonical_doc_id"],
                len(attempt_history),
                (last_quality or {}).get("failures", []),
            )
            return {
                "status": "FAILED",
                "message": f"Quality checks failed after {len(attempt_history)} attempts",
                "error_code": "QUALITY_CHECK_FAILED",
                "failed_stage": "quality_check",
                "retryable": False,
                "exc_type": "ApplicationError",
                "exc_message": f"Quality checks failed after {len(attempt_history)} attempts",
                "article_id": article_id,
                "generated_article_id": str(persisted_id),
                "canonical_doc_id": validated_submit_info["canonical_doc_id"],
                "difficulty_level": validated_submit_info["difficulty_level"],
                "attempt_count": len(attempt_history),
                "max_attempts": self.MAX_REPAIR_ATTEMPTS,
                "qa_required": True,
                "quality_summary": self.quality_checker.build_quality_summary(last_quality),
            }
        except ArticleGenerationError as exc:
            if validated_submit_info and source_context and (attempt_history or last_rendered["article_with_claim_ids"] or last_body_sections):
                try:
                    article_mongo = MongoDBHandler(
                        self.cfg.generated_article_mongo_db_name,
                        self.cfg.generated_article_collection_name,
                    )
                    persisted_id = self._persist_article_document(
                        self._build_article_document(
                            article_id=article_id,
                            submit_info=validated_submit_info,
                            settings=settings or {},
                            canonical_doc=canonical_doc or {},
                            source_context=source_context,
                            outline=outline or {"subtopics": []},
                            raw_retrieval=raw_retrieval or {"sections": []},
                            retrieval=retrieval or {"sections": []},
                            body_sections=last_body_sections,
                            intro_conclusion=last_intro_conclusion,
                            rendered=last_rendered,
                            quality=last_quality or self.quality_checker.build_exception_quality(exc.message),
                            attempt_history=attempt_history,
                            status="failed",
                            final_error_code=exc.error_code,
                            final_failed_stage=exc.failed_stage,
                            total_runtime_seconds=round(time.time() - started_at, 3),
                            embedding_doc_id=embedding_doc_id,
                        ),
                        article_mongo,
                    )
                    failure_dict = self._failure_dict(exc)
                    failure_dict.update({
                        "article_id": article_id,
                        "generated_article_id": str(persisted_id),
                        "attempt_count": len(attempt_history),
                        "max_attempts": self.MAX_REPAIR_ATTEMPTS,
                        "qa_required": True,
                        "quality_summary": self.quality_checker.build_quality_summary(last_quality or self.quality_checker.build_exception_quality(exc.message)),
                    })
                    return failure_dict
                except ArticleGenerationError as persist_exc:
                    exc = persist_exc
            logger.error(
                "generate_article failed: canonical_doc_id=%s stage=%s error_code=%s retryable=%s message=%s",
                submit_info.get("canonical_doc_id"),
                exc.failed_stage,
                exc.error_code,
                exc.retryable,
                exc.message,
            )
            return self._failure_dict(exc)
        except Exception as exc:
            logger.exception(
                "generate_article unexpected failure: canonical_doc_id=%s",
                submit_info.get("canonical_doc_id"),
            )
            wrapped = ArticleGenerationError(
                str(exc),
                error_code="UNEXPECTED_ERROR",
                failed_stage="unexpected",
                retryable=False,
            )
            return self._failure_dict(wrapped)

    def _validate_submit_info(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["user_id", "canonical_doc_id", "difficulty_level"]
        missing_fields = [field for field in required_fields if not submit_info.get(field)]
        if missing_fields:
            raise ArticleGenerationError(
                f"Missing required submit_info fields: {', '.join(missing_fields)}",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )

        difficulty_level = str(submit_info.get("difficulty_level", "")).strip().lower()
        if difficulty_level not in {"beginner", "intermediate", "expert"}:
            raise ArticleGenerationError(
                "difficulty_level must be one of: beginner, intermediate, expert",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )

        canonical_doc_id = str(submit_info.get("canonical_doc_id", "")).strip()
        if not ObjectId.is_valid(canonical_doc_id):
            raise ArticleGenerationError(
                "canonical_doc_id must be a valid Mongo ObjectId",
                error_code="VALIDATION_ERROR",
                failed_stage="validating_input",
                retryable=False,
                exc_type="ValueError",
            )

        default_word_count = {"beginner": 350, "intermediate": 800, "expert": 1500}[difficulty_level]
        raw_word_count = submit_info.get("word_count")
        if raw_word_count is None:
            word_count = default_word_count
        else:
            try:
                word_count = int(raw_word_count)
            except (TypeError, ValueError):
                raise ArticleGenerationError(
                    "word_count must be an integer",
                    error_code="VALIDATION_ERROR",
                    failed_stage="validating_input",
                    retryable=False,
                    exc_type="ValueError",
                )
            if not (350 <= word_count <= 2500):
                raise ArticleGenerationError(
                    "word_count must be between 350 and 2500",
                    error_code="VALIDATION_ERROR",
                    failed_stage="validating_input",
                    retryable=False,
                    exc_type="ValueError",
                )

        return {
            "submit_type": "generate_article",
            "user_id": str(submit_info["user_id"]).strip(),
            "canonical_doc_id": canonical_doc_id,
            "difficulty_level": difficulty_level,
            "personal_remarks": str(submit_info.get("personal_remarks", "")).strip(),
            "word_count": word_count,
        }

    def _resolve_settings(self, submit_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "outline_model": self.cfg.openai_article_outline_model,
            "writer_model": self.cfg.openai_article_writer_model,
            "intro_model": self.cfg.openai_article_intro_model,
            "embedding_model": self.cfg.openai_article_embedding_model,
            "qa_model": self.cfg.openai_article_outline_model,
            "top_k_per_point": 6,
            "near_duplicate_threshold": 0.9,
            "max_claims_per_section": 6,
            "target_word_count": submit_info["word_count"],
            "max_sources_per_claim_in_prompt": 3,
        }

    def _load_canonical_document(self, canonical_doc_id: str, mongoio: MongoDBHandler) -> Dict[str, Any]:
        object_id = ObjectId(canonical_doc_id)

        for attempt in range(1, 4):
            try:
                document = mongoio.collection.find_one({"_id": object_id})
                break
            except Exception as exc:
                self.usage["retries"] += 1
                if attempt == 3:
                    raise ArticleGenerationError(
                        f"Failed to load canonical document: {exc}",
                        error_code="MONGO_ERROR",
                        failed_stage="loading_canonical_document",
                        retryable=True,
                    ) from exc
                time.sleep(1)
        else:
            document = None

        if not document:
            raise ArticleGenerationError(
                "Canonical document not found",
                error_code="SOURCE_NOT_FOUND",
                failed_stage="loading_canonical_document",
                retryable=False,
                exc_type="ValueError",
            )

        if document.get("meta", {}).get("type") != "canonical_topic":
            raise ArticleGenerationError(
                "Source document is not a canonical_topic document",
                error_code="SOURCE_INVALID",
                failed_stage="loading_canonical_document",
                retryable=False,
                exc_type="ValueError",
            )

        return document

    def _build_source_context(self, canonical_doc: Dict[str, Any], embedding_model: str) -> Dict[str, Any]:
        topic = self.deep_get(canonical_doc, ["document", "topic"], "").strip()
        canonical_summary = self.deep_get(canonical_doc, ["document", "canonical_summary"], "").strip()
        subtopics = self.deep_get(canonical_doc, ["document", "metadata", "subtopics"], [])

        if not subtopics:
            sections = self.deep_get(canonical_doc, ["document", "sections"], [])
            subtopics = [section.get("subtopic", "").strip() for section in sections if section.get("subtopic")]

        if not topic or not canonical_summary or not subtopics:
            raise ArticleGenerationError(
                "Canonical document is missing topic, canonical_summary, or subtopics",
                error_code="SOURCE_INVALID",
                failed_stage="extracting_claim_bank",
                retryable=False,
            )

        claims = self.deep_get(canonical_doc, ["intermediate", "embedded_verified_claims", "verified_claims"], [])
        if not claims:
            claims = self.recursively_find_claims_with_embeddings(canonical_doc)

        if not claims:
            claims = self.deep_get(canonical_doc, ["intermediate", "verified_claims", "verified_claims"], [])
            if not claims:
                claims = self.recursively_find_verified_claims_without_embeddings(canonical_doc)

            texts = [self.claim_text(claim) for claim in claims if self.claim_text(claim)]
            if not texts:
                raise ArticleGenerationError(
                    "Canonical document contains no usable claims",
                    error_code="SOURCE_INVALID",
                    failed_stage="extracting_claim_bank",
                    retryable=False,
                )
            embeddings = self.embed_texts(texts, embedding_model, "extracting_claim_bank")
            emb_index = 0
            for claim in claims:
                if self.claim_text(claim):
                    claim["embedding"] = embeddings[emb_index]
                    claim["embedding_model"] = embedding_model
                    emb_index += 1

        url_to_sid, self.sid_to_url = self.build_source_registry(canonical_doc, claims)
        self.claim_bank = self.build_claim_bank(claims, url_to_sid, embedding_model)
        self.claim_id_to_claim = {claim["claim_id"]: claim for claim in self.claim_bank}

        if not self.claim_bank:
            raise ArticleGenerationError(
                "Canonical document contains no embedded claim bank",
                error_code="SOURCE_INVALID",
                failed_stage="extracting_claim_bank",
                retryable=False,
            )

        self.quality_checker = ArticleQualityChecker(self.claim_id_to_claim, self.sid_to_url, self.claim_bank)

        return {
            "topic": topic,
            "canonical_summary": canonical_summary,
            "subtopics": subtopics,
            "source_doc_id": str(canonical_doc["_id"]),
            "source_meta": canonical_doc.get("meta", {}),
        }

    def generate_outline(
        self,
        topic: str,
        canonical_summary: str,
        subtopics: List[str],
        difficulty_level: str,
        personal_remarks: str,
        model: str,
    ) -> Dict[str, Any]:
        prompt = f"""
You create structured article outlines from canonical documents.

Topic:
{topic}

Canonical summary:
{canonical_summary}

Subtopics:
{json.dumps(subtopics, ensure_ascii=False, indent=2)}

Reader difficulty:
{difficulty_level}

Personal remarks:
{personal_remarks or "None"}

Rules:
- Use the given subtopics exactly once each.
- Make sections mutually exclusive.
- Produce 3 to 5 retrieval points per subtopic.
- Match the difficulty level.
- Reflect personal remarks only in framing, examples, and emphasis.
- Do not invent facts outside the canonical summary and subtopics.

Return JSON only:
{{
  "subtopics": [
    {{
      "subtopic": "Original subtopic",
      "section_intent": "One sentence",
      "points": [
        {{"id": "P1", "intent": "Specific retrieval intent"}}
      ]
    }}
  ]
}}
"""
        result = self.ask_openai_json(prompt, model=model, stage_name="generating_outline")

        if not isinstance(result, dict) or not isinstance(result.get("subtopics"), list):
            raise ArticleGenerationError(
                "Outline must contain a subtopics array",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="generating_outline",
                retryable=False,
            )

        normalized_subtopics = []
        for section in result["subtopics"]:
            points = []
            for index, point in enumerate(section.get("points", []), start=1):
                intent = str(point.get("intent", "")).strip()
                if not intent:
                    continue
                points.append({"id": point.get("id") or f"P{index}", "intent": intent})

            if not section.get("subtopic") or not section.get("section_intent") or not points:
                continue

            normalized_subtopics.append({
                "subtopic": str(section["subtopic"]).strip(),
                "section_intent": str(section["section_intent"]).strip(),
                "points": points[:5],
            })

        if not normalized_subtopics:
            raise ArticleGenerationError(
                "Outline generation returned no usable sections",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="generating_outline",
                retryable=False,
            )

        return {"subtopics": normalized_subtopics}

    def retrieve_claims_for_outline(
        self,
        outline: Dict[str, Any],
        claim_bank: List[Dict[str, Any]],
        difficulty_level: str,
        personal_remarks: str,
        embedding_model: str,
        top_k: int,
    ) -> Dict[str, Any]:
        all_points = []
        for section in outline["subtopics"]:
            for point in section["points"]:
                query = self.build_personalized_query(point["intent"], difficulty_level, personal_remarks)
                all_points.append({
                    "subtopic": section["subtopic"],
                    "point_id": point["id"],
                    "point_intent": point["intent"],
                    "query": query,
                })

        query_embeddings = self.embed_texts(
            [point["query"] for point in all_points],
            embedding_model,
            "retrieving_claims",
        )

        retrieval = {"sections": []}
        point_idx = 0
        for section in outline["subtopics"]:
            point_hits = []
            for point in section["points"]:
                scores = []
                query_embedding = query_embeddings[point_idx]
                for claim in claim_bank:
                    scores.append(self.cosine_similarity(query_embedding, claim["embedding"]))

                top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
                hits = []
                for idx in top_indices:
                    claim = dict(claim_bank[idx])
                    claim["similarity"] = float(scores[idx])
                    claim["supports_point_ids"] = [point["id"]]
                    hits.append(claim)

                point_hits.append({
                    "point_id": point["id"],
                    "point_intent": point["intent"],
                    "hits": hits,
                })
                point_idx += 1

            retrieval["sections"].append({
                "subtopic": section["subtopic"],
                "section_intent": section["section_intent"],
                "points": section["points"],
                "point_hits": point_hits,
            })

        return retrieval

    def merge_and_filter_retrieval(
        self,
        raw_retrieval: Dict[str, Any],
        near_duplicate_threshold: float,
        max_claims_per_section: int,
    ) -> Dict[str, Any]:
        merged = {"sections": []}

        for section in raw_retrieval["sections"]:
            by_claim_id: Dict[str, Dict[str, Any]] = {}
            for point_hit in section["point_hits"]:
                point_id = point_hit["point_id"]

                for hit in point_hit["hits"]:
                    if not self.quality_pass(hit):
                        continue

                    claim_id = hit["claim_id"]
                    if claim_id not in by_claim_id:
                        by_claim_id[claim_id] = dict(hit)
                        by_claim_id[claim_id]["supports_point_ids"] = [point_id]
                        by_claim_id[claim_id]["similarities"] = [hit["similarity"]]
                    else:
                        by_claim_id[claim_id]["supports_point_ids"].append(point_id)
                        by_claim_id[claim_id]["similarities"].append(hit["similarity"])
                        by_claim_id[claim_id]["similarity"] = max(
                            by_claim_id[claim_id]["similarity"],
                            hit["similarity"],
                        )

            candidates = list(by_claim_id.values())
            for candidate in candidates:
                candidate["supports_point_ids"] = sorted(set(candidate["supports_point_ids"]))
                sims = candidate.get("similarities", [candidate.get("similarity", 0.0)])
                candidate["mean_similarity"] = round(sum(sims) / max(len(sims), 1), 6)

            candidates = self.collapse_near_duplicates(candidates, near_duplicate_threshold)
            candidates = sorted(
                candidates,
                key=lambda item: (
                    len(item.get("supports_point_ids", [])),
                    self.trust_rank(item.get("trust_label", "unknown")),
                    float(item.get("weighted_support_score", 0.0)),
                    float(item.get("similarity", 0.0)),
                ),
                reverse=True,
            )
            if max_claims_per_section > 0:
                candidates = candidates[:max_claims_per_section]

            merged["sections"].append({
                "subtopic": section["subtopic"],
                "section_intent": section["section_intent"],
                "points": section["points"],
                "claims": candidates,
            })

        return merged

    def write_body_sections(
        self,
        retrieval: Dict[str, Any],
        outline: Dict[str, Any],
        difficulty_level: str,
        personal_remarks: str,
        writer_model: str,
        target_word_count: int,
    ) -> List[Dict[str, Any]]:
        section_count = max(len(retrieval["sections"]), 1)
        target_words_per_section = max(100, target_word_count // section_count)
        full_coverage_map = self.coverage_map_text(outline)
        body_sections = []
        previous_text = ""
        already_covered = []

        for section in retrieval["sections"]:
            prompt = f"""
You write one body section of a larger article.

Section:
{section["subtopic"]}

Section intent:
{section["section_intent"]}

Section points:
{self.format_points(section["points"])}

Coverage map:
{full_coverage_map}

Difficulty level:
{difficulty_level}

Personal remarks:
{personal_remarks or "None"}

Previously covered:
{chr(10).join(already_covered) or "Nothing yet."}

Previous section tail:
{self.last_n_sentences(previous_text, 3) if previous_text else "This is the first section."}

Claims:
{self.format_claim_lines(section["claims"])}

Rules:
- Write only the body for this section.
- Do not write an article introduction or conclusion.
- Match the difficulty level.
- Reflect personal remarks only in framing and emphasis.
- Every factual sentence must end with one or more claim tags like [C0001].
- If a sentence has no claim tag, it must be pure transition or interpretation.
- Do not invent facts.
- Target about {target_words_per_section} words.
"""
            body = self.ask_openai_text(prompt, model=writer_model, stage_name="writing_sections", temperature=0.35)
            body_sections.append({
                "subtopic": section["subtopic"],
                "body": body.strip(),
                "claims_used_in_prompt": [claim["claim_id"] for claim in section["claims"]],
            })
            previous_text = body
            already_covered.append(f"- {section['subtopic']}: {section['section_intent']}")

        return body_sections

    def write_intro_and_conclusion(
        self,
        topic: str,
        canonical_summary: str,
        body_sections: List[Dict[str, Any]],
        difficulty_level: str,
        personal_remarks: str,
        model: str,
    ) -> Dict[str, str]:
        draft_body = self.body_draft_markdown(body_sections)
        cited_claim_ids = self.extract_claim_ids(draft_body)

        prompt = f"""
You write the introduction and conclusion for an article that already has body sections.

Topic:
{topic}

Canonical summary:
{canonical_summary}

Difficulty level:
{difficulty_level}

Personal remarks:
{personal_remarks or "None"}

Body draft:
{draft_body}

Allowed claims:
{self.claim_lines_by_ids(cited_claim_ids)}

Rules:
- Do not add new facts.
- If a sentence is factual, it must end with claim tags.
- Use only the allowed claims.
- Return JSON only:
{{
  "introduction": "text",
  "conclusion": "text"
}}
"""
        result = self.ask_openai_json(prompt, model=model, stage_name="writing_intro_conclusion", temperature=0.3)
        if not isinstance(result, dict):
            raise ArticleGenerationError(
                "Intro/conclusion writer must return an object",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="writing_intro_conclusion",
                retryable=False,
            )

        introduction = str(result.get("introduction", "")).strip()
        conclusion = str(result.get("conclusion", "")).strip()
        if not introduction or not conclusion:
            raise ArticleGenerationError(
                "Intro/conclusion writer returned empty text",
                error_code="OPENAI_JSON_ERROR",
                failed_stage="writing_intro_conclusion",
                retryable=False,
            )

        return {"introduction": introduction, "conclusion": conclusion}

    def render_article(
        self,
        topic: str,
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
    ) -> Dict[str, Any]:
        draft_body = self.body_draft_markdown(body_sections)
        article_with_claim_ids = (
            f"# {topic}\n\n"
            f"{intro_conclusion['introduction'].strip()}\n\n"
            f"{draft_body}\n\n"
            f"## Conclusion\n\n"
            f"{intro_conclusion['conclusion'].strip()}\n"
        )
        article_with_sources, used_source_ids = self.render_article_with_source_footnotes(article_with_claim_ids)
        return {
            "article_with_claim_ids": article_with_claim_ids,
            "article_with_sources": article_with_sources,
            "used_source_ids": sorted(used_source_ids),
        }

    def repair_article_attempt(
        self,
        retrieval: Dict[str, Any],
        outline: Dict[str, Any],
        topic: str,
        canonical_summary: str,
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        quality: Dict[str, Any],
        difficulty_level: str,
        personal_remarks: str,
        writer_model: str,
        intro_model: str,
        target_word_count: int,
        attempt_number: int,
    ) -> Dict[str, Any]:
        instructions = quality.get("repair_instructions", {})
        targeted_sections = set(instructions.get("section_targets", []))
        regenerate_all_sections = instructions.get("regenerate_all_sections", False)
        regenerate_intro = instructions.get("repair_intro", False)
        regenerate_conclusion = instructions.get("repair_conclusion", False)
        full_coverage_map = self.coverage_map_text(outline)
        section_count = max(len(retrieval["sections"]), 1)
        target_words_per_section = max(100, target_word_count // section_count)

        updated_sections = []
        previous_text = ""
        for section in retrieval["sections"]:
            current_section = next(
                (item for item in body_sections if item["subtopic"] == section["subtopic"]),
                {"subtopic": section["subtopic"], "body": "", "claims_used_in_prompt": []},
            )
            should_repair = regenerate_all_sections or section["subtopic"] in targeted_sections
            if should_repair:
                findings = [
                    finding for finding in quality.get("findings", [])
                    if finding.get("subtopic") == section["subtopic"]
                ]
                repaired_body = self.rewrite_section_for_quality(
                    section=section,
                    existing_body=current_section["body"],
                    findings=findings,
                    full_coverage_map=full_coverage_map,
                    difficulty_level=difficulty_level,
                    personal_remarks=personal_remarks,
                    previous_section_text=previous_text,
                    writer_model=writer_model,
                    target_words_per_section=target_words_per_section,
                    attempt_number=attempt_number,
                )
                updated_sections.append({
                    "subtopic": section["subtopic"],
                    "body": repaired_body.strip(),
                    "claims_used_in_prompt": [claim["claim_id"] for claim in section["claims"]],
                })
                previous_text = repaired_body
            else:
                updated_sections.append(current_section)
                previous_text = current_section["body"]

        updated_intro_conclusion = dict(intro_conclusion)
        if regenerate_intro or regenerate_conclusion or regenerate_all_sections or targeted_sections:
            updated_intro_conclusion = self.rewrite_intro_and_conclusion_for_quality(
                topic=topic,
                canonical_summary=canonical_summary,
                body_sections=updated_sections,
                current_intro_conclusion=intro_conclusion,
                quality=quality,
                difficulty_level=difficulty_level,
                personal_remarks=personal_remarks,
                model=intro_model,
                rewrite_intro=regenerate_intro or regenerate_all_sections,
                rewrite_conclusion=regenerate_conclusion or regenerate_all_sections,
            )

        return {
            "body_sections": updated_sections,
            "intro_conclusion": updated_intro_conclusion,
            "repair_instructions": instructions,
        }

    def rewrite_section_for_quality(
        self,
        section: Dict[str, Any],
        existing_body: str,
        findings: List[Dict[str, Any]],
        full_coverage_map: str,
        difficulty_level: str,
        personal_remarks: str,
        previous_section_text: str,
        writer_model: str,
        target_words_per_section: int,
        attempt_number: int,
    ) -> str:
        failing_sentences = "\n".join(
            [f"- {finding.get('sentence')}" for finding in findings if finding.get("sentence")]
        ) or "No sentence-level findings provided."
        prompt = f"""
You are repairing one article section that failed quality checks.

Attempt number:
{attempt_number}

Section:
{section["subtopic"]}

Section intent:
{section["section_intent"]}

Section points:
{self.format_points(section["points"])}

Coverage map:
{full_coverage_map}

Current section draft:
{existing_body}

Flagged content:
{failing_sentences}

Difficulty level:
{difficulty_level}

Personal remarks:
{personal_remarks or "None"}

Previous section tail:
{self.last_n_sentences(previous_section_text, 3) if previous_section_text else "This is the first section."}

Allowed claims:
{self.format_claim_lines(section["claims"])}

Rules:
- Rewrite only this section.
- Preserve correct material when possible.
- Remove unsupported sentences if needed.
- Every factual sentence must end with claim tags like [C0001].
- Use only the allowed claims.
- Do not add any unsupported facts.
- Keep the section close to {target_words_per_section} words.
"""
        return self.ask_openai_text(
            prompt,
            model=writer_model,
            stage_name="repairing_article",
            temperature=0.2,
        )

    def rewrite_intro_and_conclusion_for_quality(
        self,
        topic: str,
        canonical_summary: str,
        body_sections: List[Dict[str, Any]],
        current_intro_conclusion: Dict[str, str],
        quality: Dict[str, Any],
        difficulty_level: str,
        personal_remarks: str,
        model: str,
        rewrite_intro: bool,
        rewrite_conclusion: bool,
    ) -> Dict[str, str]:
        cited_claim_ids = self.extract_claim_ids(self.body_draft_markdown(body_sections))
        intro_findings = [
            finding for finding in quality.get("findings", [])
            if finding.get("location") == "introduction"
        ]
        conclusion_findings = [
            finding for finding in quality.get("findings", [])
            if finding.get("location") == "conclusion"
        ]
        prompt = f"""
You are repairing the introduction and conclusion of an article after failed quality checks.

Topic:
{topic}

Canonical summary:
{canonical_summary}

Difficulty level:
{difficulty_level}

Personal remarks:
{personal_remarks or "None"}

Current introduction:
{current_intro_conclusion.get("introduction", "")}

Current conclusion:
{current_intro_conclusion.get("conclusion", "")}

Body draft:
{self.body_draft_markdown(body_sections)}

Allowed claims:
{self.claim_lines_by_ids(cited_claim_ids)}

Intro findings:
{json.dumps(intro_findings, ensure_ascii=False)}

Conclusion findings:
{json.dumps(conclusion_findings, ensure_ascii=False)}

Rewrite introduction:
{str(rewrite_intro).lower()}

Rewrite conclusion:
{str(rewrite_conclusion).lower()}

Rules:
- Return JSON only with keys introduction and conclusion.
- If rewrite introduction is false, keep the current introduction meaningfully unchanged.
- If rewrite conclusion is false, keep the current conclusion meaningfully unchanged.
- Every factual sentence must end with claim tags.
- Use only the allowed claims.
- Remove unsupported factual sentences if necessary.
"""
        result = self.ask_openai_json(
            prompt,
            model=model,
            stage_name="repairing_article",
            temperature=0.2,
        )
        return {
            "introduction": str(result.get("introduction", current_intro_conclusion.get("introduction", ""))).strip(),
            "conclusion": str(result.get("conclusion", current_intro_conclusion.get("conclusion", ""))).strip(),
        }

    def run_quality_checks_with_openai(
        self,
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        rendered: Dict[str, Any],
        qa_model: str,
    ) -> Dict[str, Any]:
        untagged_findings = self.find_untagged_factual_sentences_with_openai(
            retrieval=retrieval,
            body_sections=body_sections,
            intro_conclusion=intro_conclusion,
            model=qa_model,
        )
        return self.quality_checker.run_quality_checks(
            retrieval=retrieval,
            body_sections=body_sections,
            intro_conclusion=intro_conclusion,
            article_with_claim_ids=rendered["article_with_claim_ids"],
            article_with_sources=rendered["article_with_sources"],
            untagged_findings=untagged_findings,
        )

    def find_untagged_factual_sentences_with_openai(
        self,
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        model: str,
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        intro_claim_ids = self.extract_claim_ids(self.body_draft_markdown(body_sections))

        findings.extend(
            self.inspect_text_block_for_untagged_factual_sentences(
                text=intro_conclusion.get("introduction", ""),
                location="introduction",
                allowed_claim_lines=self.claim_lines_by_ids(intro_claim_ids),
                model=model,
            )
        )

        for section in body_sections:
            retrieval_section = next(
                (item for item in retrieval["sections"] if item["subtopic"] == section["subtopic"]),
                {"claims": []},
            )
            findings.extend(
                self.inspect_text_block_for_untagged_factual_sentences(
                    text=section.get("body", ""),
                    location=f"section:{section['subtopic']}",
                    subtopic=section["subtopic"],
                    allowed_claim_lines=self.format_claim_lines(retrieval_section.get("claims", [])),
                    model=model,
                )
            )

        findings.extend(
            self.inspect_text_block_for_untagged_factual_sentences(
                text=intro_conclusion.get("conclusion", ""),
                location="conclusion",
                allowed_claim_lines=self.claim_lines_by_ids(intro_claim_ids),
                model=model,
            )
        )
        return findings

    def inspect_text_block_for_untagged_factual_sentences(
        self,
        text: str,
        location: str,
        allowed_claim_lines: str,
        model: str,
        subtopic: str | None = None,
    ) -> List[Dict[str, Any]]:
        paragraphs = self.split_paragraphs(text)
        findings: List[Dict[str, Any]] = []

        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            prompt = f"""
You are checking one article paragraph for unsupported inline citation formatting.

Paragraph location:
{location}

Allowed claims:
{allowed_claim_lines or "None"}

Paragraph:
{paragraph}

Rules:
- A factual sentence must end with one or more claim tags like [C0001].
- Transitional, interpretive, or stylistic sentences do not need claim tags.
- Do not flag sentences that already contain claim tags.
- Be conservative. Only flag a sentence if it makes a concrete factual claim and lacks claim tags.

Return JSON only:
{{
  "has_untagged_factual_sentences": true,
  "sentences": ["exact sentence from the paragraph"]
}}
"""
            result = self.ask_openai_json(
                prompt,
                model=model,
                stage_name="quality_check",
                temperature=0.0,
            )
            for sentence in result.get("sentences", []):
                cleaned = str(sentence).strip()
                if not cleaned or self.CLAIM_TAG_RE.search(cleaned):
                    continue
                finding = {
                    "check": "untagged_factual_sentences",
                    "location": location,
                    "subtopic": subtopic,
                    "paragraph_index": paragraph_index,
                    "sentence": cleaned,
                    "suggested_action": "rewrite_sentence_with_claim_tags" if location in {"introduction", "conclusion"} else "regenerate_section",
                }
                if finding not in findings:
                    findings.append(finding)

        return findings

    def repair_untagged_paragraphs_with_openai(
        self,
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        findings: List[Dict[str, Any]],
        writer_model: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        untagged_findings = [f for f in findings if f.get("check") == "untagged_factual_sentences"]
        if not untagged_findings:
            return body_sections, intro_conclusion

        grouped_findings: Dict[str, List[Dict[str, Any]]] = {}
        for finding in untagged_findings:
            grouped_findings.setdefault(finding["location"], []).append(finding)

        body_claim_ids = self.extract_claim_ids(self.body_draft_markdown(body_sections))
        updated_intro_conclusion = dict(intro_conclusion)
        updated_sections: List[Dict[str, Any]] = []

        if "introduction" in grouped_findings:
            updated_intro_conclusion["introduction"] = self.rewrite_text_block_to_remove_untagged_factual_sentences(
                text=intro_conclusion.get("introduction", ""),
                location="introduction",
                flagged_findings=grouped_findings["introduction"],
                allowed_claim_lines=self.claim_lines_by_ids(body_claim_ids),
                model=writer_model,
            )

        if "conclusion" in grouped_findings:
            updated_intro_conclusion["conclusion"] = self.rewrite_text_block_to_remove_untagged_factual_sentences(
                text=intro_conclusion.get("conclusion", ""),
                location="conclusion",
                flagged_findings=grouped_findings["conclusion"],
                allowed_claim_lines=self.claim_lines_by_ids(body_claim_ids),
                model=writer_model,
            )

        for section in body_sections:
            location = f"section:{section['subtopic']}"
            if location not in grouped_findings:
                updated_sections.append(section)
                continue

            retrieval_section = next(
                (item for item in retrieval["sections"] if item["subtopic"] == section["subtopic"]),
                {"claims": []},
            )
            updated_sections.append({
                **section,
                "body": self.rewrite_text_block_to_remove_untagged_factual_sentences(
                    text=section.get("body", ""),
                    location=location,
                    flagged_findings=grouped_findings[location],
                    allowed_claim_lines=self.format_claim_lines(retrieval_section.get("claims", [])),
                    model=writer_model,
                ),
            })

        return updated_sections, updated_intro_conclusion

    def rewrite_text_block_to_remove_untagged_factual_sentences(
        self,
        text: str,
        location: str,
        flagged_findings: List[Dict[str, Any]],
        allowed_claim_lines: str,
        model: str,
    ) -> str:
        prompt = f"""
You are repairing one article text block after QA flagged uncited factual sentences.

Block location:
{location}

Current text:
{text}

Flagged sentences:
{json.dumps([finding.get("sentence", "") for finding in flagged_findings], ensure_ascii=False)}

Allowed claims:
{allowed_claim_lines or "None"}

Rules:
- Rewrite the full text block.
- Every factual sentence must end with one or more claim tags like [C0001].
- Use only the allowed claims.
- Do not add new facts.
- If a factual sentence cannot be supported by the allowed claims, remove it or convert it into non-factual transition text.
- Preserve the original meaning when the claims support it.
- Return plain text only. No markdown fences.
"""
        return self.ask_openai_text(
            prompt,
            model=model,
            stage_name="repairing_article",
            temperature=0.1,
        ).strip()

    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]

    def _build_article_document(
        self,
        article_id: str,
        submit_info: Dict[str, Any],
        settings: Dict[str, Any],
        canonical_doc: Dict[str, Any],
        source_context: Dict[str, Any],
        outline: Dict[str, Any],
        raw_retrieval: Dict[str, Any],
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        rendered: Dict[str, Any],
        quality: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
        status: str,
        final_error_code: str | None,
        final_failed_stage: str | None,
        total_runtime_seconds: float,
        embedding_doc_id: str | None = None,
    ) -> Dict[str, Any]:
        created_at = datetime.utcnow().isoformat()
        quality = quality or self.quality_checker.build_exception_quality("No quality information available")
        failed = status == "failed"
        return {
            "meta": {
                "article_id": article_id,
                "canonical_doc_id": canonical_doc.get("meta", {}).get("doc_id") if canonical_doc else None,
                "canonical_source_mongo_id": source_context["source_doc_id"],
                "user_id": submit_info["user_id"],
                "type": "generated_article",
                "status": status,
                "difficulty_level": submit_info["difficulty_level"],
                "pipeline_version": "v1_generate_article",
                "creation_date_human": datetime.now().strftime("%B %d, %Y, %I:%M %p"),
                "creation_time": time.time(),
                "created_at": created_at,
                "attempt_count": len(attempt_history),
                "max_attempts": self.MAX_REPAIR_ATTEMPTS,
                "qa_required": failed,
                "repair_exhausted": failed and len(attempt_history) >= self.MAX_REPAIR_ATTEMPTS,
                "final_error_code": final_error_code,
                "final_failed_stage": final_failed_stage,
                "failure_reasons": quality.get("failures", []) if failed else [],
                "failure_message": quality.get("message") if failed else None,
                "embedding_doc_id": embedding_doc_id,
            },
            "input": {
                "submit_info": submit_info,
                "effective_settings": settings,
            },
            "source": {
                "topic": source_context["topic"],
                "canonical_summary": source_context["canonical_summary"],
                "subtopics": source_context["subtopics"],
                "source_meta": source_context["source_meta"],
            },
            "document": {
                "title": source_context["topic"],
                "introduction": intro_conclusion["introduction"],
                "sections": body_sections,
                "conclusion": intro_conclusion["conclusion"],
                "article_with_claim_ids": rendered["article_with_claim_ids"],
                "article_with_sources": rendered["article_with_sources"],
            },
            "intermediate": {
                "outline": outline,
                "raw_retrieval": self._strip_embeddings_from_raw_retrieval(raw_retrieval),
                "filtered_retrieval": self._strip_embeddings_from_filtered_retrieval(retrieval),
                "body_sections": body_sections,
                "intro_conclusion": intro_conclusion,
                "used_source_ids": rendered["used_source_ids"],
                "attempts": attempt_history,
            },
            "quality": {
                **quality,
                "attempt_failures": [
                    {
                        "attempt_number": attempt["attempt_number"],
                        "failures": attempt["quality"].get("failures", []),
                        "passed": attempt["quality"].get("passed", False),
                    }
                    for attempt in attempt_history
                ],
                "qa_notes": [],
            },
            "usage": {
                **self.usage,
                "total_runtime_seconds": total_runtime_seconds,
            },
        }

    def _persist_article_document(self, article_doc: Dict[str, Any], mongoio: MongoDBHandler):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return mongoio.write_document(article_doc)
            except Exception as exc:
                self.usage["retries"] += 1
                last_exc = exc
                if attempt == 3:
                    raise ArticleGenerationError(
                        f"Failed to persist article document: {exc}",
                        error_code="MONGO_ERROR",
                        failed_stage="persisting_document",
                        retryable=True,
                    ) from exc
                time.sleep(1)
        raise ArticleGenerationError(
            f"Failed to persist article document: {last_exc}",
            error_code="MONGO_ERROR",
            failed_stage="persisting_document",
            retryable=True,
        )

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
                                "You are a careful article-generation assistant. "
                                "Return valid JSON only. No markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                self.usage["json_calls"] += 1
                self._merge_usage_bucket("json_usage", getattr(response, "usage", None))

                content = response.choices[0].message.content or ""
                last_content = content
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    last_error = exc
                    match = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
            except BadRequestError as exc:
                raise ArticleGenerationError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                self.usage["retries"] += 1
                last_error = exc
                if attempt == max_retries:
                    raise ArticleGenerationError(
                        self._format_openai_error(exc),
                        error_code="OPENAI_API_ERROR",
                        failed_stage=stage_name,
                        retryable=True,
                    ) from exc
                time.sleep(1)
            except APIError as exc:
                raise ArticleGenerationError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc

        raise ArticleGenerationError(
            (
                f"OpenAI failed to return valid JSON after {max_retries} attempts. "
                f"Last error: {last_error}. Last content: {last_content}"
            ),
            error_code="OPENAI_JSON_ERROR",
            failed_stage=stage_name,
            retryable=False,
        )

    def ask_openai_text(
        self,
        prompt: str,
        model: str,
        stage_name: str,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful article-generation assistant. "
                                "Follow citation instructions exactly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                self.usage["text_calls"] += 1
                self._merge_usage_bucket("text_usage", getattr(response, "usage", None))
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content.strip()
                raise ValueError("OpenAI returned empty text.")
            except BadRequestError as exc:
                raise ArticleGenerationError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                self.usage["retries"] += 1
                if attempt == max_retries:
                    raise ArticleGenerationError(
                        self._format_openai_error(exc),
                        error_code="OPENAI_API_ERROR",
                        failed_stage=stage_name,
                        retryable=True,
                    ) from exc
                time.sleep(1)
            except APIError as exc:
                raise ArticleGenerationError(
                    self._format_openai_error(exc),
                    error_code="OPENAI_API_ERROR",
                    failed_stage=stage_name,
                    retryable=False,
                ) from exc
            except Exception as exc:
                if attempt == max_retries:
                    raise ArticleGenerationError(
                        str(exc),
                        error_code="OPENAI_TEXT_ERROR",
                        failed_stage=stage_name,
                        retryable=False,
                    ) from exc
                time.sleep(1)

        raise ArticleGenerationError(
            "OpenAI failed to return usable text",
            error_code="OPENAI_TEXT_ERROR",
            failed_stage=stage_name,
            retryable=False,
        )

    def embed_texts(
        self,
        texts: List[str],
        model: str,
        stage_name: str,
        batch_size: int = 64,
    ) -> List[List[float]]:
        all_embeddings = []
        for index in range(0, len(texts), batch_size):
            batch = texts[index:index + batch_size]
            for attempt in range(1, 4):
                try:
                    response = self.client.embeddings.create(model=model, input=batch)
                    self.usage["embedding_calls"] += 1
                    self._merge_usage_bucket("embedding_usage", getattr(response, "usage", None))
                    all_embeddings.extend([item.embedding for item in response.data])
                    break
                except BadRequestError as exc:
                    raise ArticleGenerationError(
                        self._format_openai_error(exc),
                        error_code="EMBEDDING_ERROR",
                        failed_stage=stage_name,
                        retryable=False,
                    ) from exc
                except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                    self.usage["retries"] += 1
                    if attempt == 3:
                        raise ArticleGenerationError(
                            self._format_openai_error(exc),
                            error_code="EMBEDDING_ERROR",
                            failed_stage=stage_name,
                            retryable=True,
                        ) from exc
                    time.sleep(1)
                except APIError as exc:
                    raise ArticleGenerationError(
                        self._format_openai_error(exc),
                        error_code="EMBEDDING_ERROR",
                        failed_stage=stage_name,
                        retryable=False,
                    ) from exc
        return all_embeddings

    @staticmethod
    def deep_get(data: Dict[str, Any], path: List[str], default=None):
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def recursively_find_claims_with_embeddings(obj: Any) -> List[Dict[str, Any]]:
        found = []

        def walk(value):
            if isinstance(value, dict):
                has_text = "canonical_claim" in value or "claim" in value or "point" in value
                has_embedding = isinstance(value.get("embedding"), list)
                if has_text and has_embedding:
                    found.append(value)
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(obj)
        return found

    @staticmethod
    def recursively_find_verified_claims_without_embeddings(obj: Any) -> List[Dict[str, Any]]:
        found = []

        def walk(value):
            if isinstance(value, dict):
                has_text = "canonical_claim" in value or "claim" in value
                has_trust = "trust_label" in value
                if has_text and has_trust:
                    found.append(value)
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(obj)
        return found

    @staticmethod
    def claim_text(claim: Dict[str, Any]) -> str:
        return (
            claim.get("canonical_claim")
            or claim.get("claim")
            or claim.get("point")
            or ""
        ).strip()

    @staticmethod
    def claim_urls(claim: Dict[str, Any]) -> List[str]:
        urls = []
        for key in ["supporting_source_urls", "sources", "source_urls"]:
            value = claim.get(key)
            if isinstance(value, list):
                urls.extend([url for url in value if isinstance(url, str)])
            elif isinstance(value, str):
                urls.append(value)

        if isinstance(claim.get("source_url"), str):
            urls.append(claim["source_url"])

        output = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                output.append(url)
                seen.add(url)
        return output

    @staticmethod
    def claim_score(claim: Dict[str, Any]) -> float:
        return float(
            claim.get("weighted_support_score")
            or claim.get("source_weight")
            or claim.get("score")
            or 0.0
        )

    @staticmethod
    def trust_rank(label: str) -> int:
        return {
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 0,
            "unverified": 0,
        }.get(str(label).lower(), 0)

    def build_source_registry(
        self,
        canonical_doc: Dict[str, Any],
        claims: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        urls = []
        source_index = self.deep_get(canonical_doc, ["document", "source_index"], [])
        for item in source_index:
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])

        for claim in claims:
            urls.extend(self.claim_urls(claim))

        unique_urls = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                unique_urls.append(url)
                seen.add(url)

        url_to_sid = {url: f"S{i + 1:03d}" for i, url in enumerate(unique_urls)}
        sid_to_url = {sid: url for url, sid in url_to_sid.items()}
        return url_to_sid, sid_to_url

    def build_claim_bank(
        self,
        claims: List[Dict[str, Any]],
        url_to_sid: Dict[str, str],
        default_embedding_model: str,
    ) -> List[Dict[str, Any]]:
        bank = []
        for index, claim in enumerate(claims, start=1):
            text = self.claim_text(claim)
            embedding = claim.get("embedding")
            if not text or not isinstance(embedding, list):
                continue

            urls = self.claim_urls(claim)
            source_ids = [url_to_sid[url] for url in urls if url in url_to_sid]
            bank.append({
                "claim_id": f"C{index:04d}",
                "text": text,
                "trust_label": str(claim.get("trust_label", "unknown")).lower(),
                "weighted_support_score": self.claim_score(claim),
                "unique_source_count": int(claim.get("unique_source_count") or len(set(urls)) or 0),
                "source_urls": urls,
                "source_ids": source_ids,
                "supporting_quotes": claim.get("supporting_quotes", []),
                "embedding_model": claim.get("embedding_model", default_embedding_model),
                "embedding": embedding,
            })
        return bank

    @staticmethod
    def build_personalized_query(point_intent: str, difficulty_level: str, personal_remarks: str) -> str:
        difficulty_guidance = {
            "beginner": "Explain foundations, define terms, assume very little prior knowledge.",
            "intermediate": "Assume some familiarity, balance explanation and detail.",
            "expert": "Assume strong background, prioritize nuance, caveats, and dense explanations.",
        }[difficulty_level]
        return (
            f"{point_intent}\n"
            f"Difficulty guidance: {difficulty_guidance}\n"
            f"Personal remarks: {personal_remarks or 'None'}"
        )

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def quality_pass(claim: Dict[str, Any]) -> bool:
        if claim.get("trust_label") in {"low", "unverified"}:
            return False
        return int(claim.get("unique_source_count", 0)) >= 1

    def collapse_near_duplicates(
        self,
        candidates: List[Dict[str, Any]],
        near_duplicate_threshold: float,
    ) -> List[Dict[str, Any]]:
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                self.trust_rank(item.get("trust_label", "unknown")),
                float(item.get("weighted_support_score", 0.0)),
                float(item.get("similarity", 0.0)),
            ),
            reverse=True,
        )

        kept = []
        for candidate in sorted_candidates:
            duplicate_of = None
            for kept_item in kept:
                similarity = self.cosine_similarity(candidate["embedding"], kept_item["embedding"])
                if similarity >= near_duplicate_threshold:
                    duplicate_of = kept_item
                    break

            if duplicate_of is None:
                kept.append(candidate)
            else:
                duplicate_of["supports_point_ids"] = sorted(
                    set(duplicate_of.get("supports_point_ids", [])) | set(candidate.get("supports_point_ids", []))
                )
        return kept

    @staticmethod
    def coverage_map_text(outline: Dict[str, Any]) -> str:
        lines = []
        for index, section in enumerate(outline["subtopics"], start=1):
            intents = "; ".join(point["intent"] for point in section["points"])
            lines.append(f"{index}. {section['subtopic']}: {intents}")
        return "\n".join(lines)

    @staticmethod
    def format_points(points: List[Dict[str, str]]) -> str:
        return "\n".join([f"{point['id']}. {point['intent']}" for point in points])

    def format_claim_lines(self, claims: List[Dict[str, Any]]) -> str:
        lines = []
        for claim in claims:
            source_part = ", ".join(claim.get("source_ids", [])[:3]) or "no-source-id"
            lines.append(f"[{claim['claim_id']}] {claim['text']} [sources: {source_part}]")
        return "\n".join(lines)

    def claim_lines_by_ids(self, claim_ids: List[str]) -> str:
        lines = []
        for claim_id in claim_ids:
            claim = self.claim_id_to_claim.get(claim_id)
            if not claim:
                continue
            source_part = ", ".join(claim.get("source_ids", [])[:3]) or "no-source-id"
            lines.append(f"[{claim_id}] {claim['text']} [sources: {source_part}]")
        return "\n".join(lines)

    @staticmethod
    def sentence_split(text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        return re.split(r"(?<=[.!?])\s+", normalized)

    def last_n_sentences(self, text: str, count: int) -> str:
        sentences = self.sentence_split(text)
        return " ".join(sentences[-count:])

    @staticmethod
    def body_draft_markdown(body_sections: List[Dict[str, Any]]) -> str:
        return "\n\n".join([f"## {section['subtopic']}\n\n{section['body'].strip()}" for section in body_sections])

    def extract_claim_ids(self, text: str) -> List[str]:
        return sorted(set(self.CLAIM_TAG_RE.findall(text)))

    def render_article_with_source_footnotes(self, article: str) -> Tuple[str, set]:
        used_source_ids = set()

        def replace_claim_tag(match):
            claim_id = match.group(1)
            claim = self.claim_id_to_claim.get(claim_id)
            if not claim:
                return f"[{claim_id}]"
            source_ids = claim.get("source_ids", [])[:3]
            for source_id in source_ids:
                used_source_ids.add(source_id)
            if not source_ids:
                return f"[{claim_id}]"
            return "".join([f"[^{source_id}]" for source_id in source_ids])

        rendered = self.CLAIM_TAG_RE.sub(replace_claim_tag, article)
        if used_source_ids:
            rendered += "\n\n## Sources\n\n"
            rendered += "\n".join([
                f"[^{source_id}]: {self.sid_to_url[source_id]}"
                for source_id in sorted(used_source_ids)
                if source_id in self.sid_to_url
            ])
        return rendered, used_source_ids

    def build_attempt_record(
        self,
        attempt_number: int,
        attempt_type: str,
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        rendered: Dict[str, Any],
        quality: Dict[str, Any],
        repair_instructions: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return {
            "attempt_number": attempt_number,
            "attempt_type": attempt_type,
            "body_sections": body_sections,
            "intro_conclusion": intro_conclusion,
            "rendered": rendered,
            "quality": quality,
            "repair_instructions": repair_instructions or {},
        }

    def resolve_untagged_sentences(
        self,
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        findings: List[Dict[str, Any]],
        embedding_model: str,
        similarity_threshold: float = 0.75,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        untagged = [f for f in findings if f.get("check") == "untagged_factual_sentences"]
        if not untagged or not self.claim_bank:
            return body_sections, intro_conclusion

        unique_sentences = list({f["sentence"] for f in untagged if f.get("sentence")})
        if not unique_sentences:
            return body_sections, intro_conclusion

        sentence_embeddings = self.embed_texts(unique_sentences, embedding_model, "quality_check")

        resolved: Dict[str, str | None] = {}
        for sentence, sentence_emb in zip(unique_sentences, sentence_embeddings):
            best_score = 0.0
            best_claim_id = None
            for claim in self.claim_bank:
                score = self.cosine_similarity(sentence_emb, claim["embedding"])
                if score > best_score:
                    best_score = score
                    best_claim_id = claim["claim_id"]
            resolved[sentence] = best_claim_id if best_score >= similarity_threshold else None

        intro_sentences = {f["sentence"] for f in untagged if f.get("location") == "introduction"}
        conclusion_sentences = {f["sentence"] for f in untagged if f.get("location") == "conclusion"}
        section_sentences: Dict[str, set] = {}
        for f in untagged:
            if f.get("location", "").startswith("section:"):
                section_sentences.setdefault(f["subtopic"], set()).add(f["sentence"])

        def apply_fixes(text: str, sentences_to_fix: set) -> str:
            for sentence in sentences_to_fix:
                claim_id = resolved.get(sentence)
                if claim_id:
                    text = text.replace(sentence, self._insert_claim_tag(sentence, claim_id))
                else:
                    text = text.replace(sentence, "")
            return re.sub(r" {2,}", " ", text).strip()

        updated_intro = apply_fixes(intro_conclusion.get("introduction", ""), intro_sentences)
        updated_conclusion = apply_fixes(intro_conclusion.get("conclusion", ""), conclusion_sentences)

        updated_sections = []
        for section in body_sections:
            sentences_for_section = section_sentences.get(section["subtopic"], set())
            if sentences_for_section:
                updated_sections.append({**section, "body": apply_fixes(section["body"], sentences_for_section)})
            else:
                updated_sections.append(section)

        return updated_sections, {**intro_conclusion, "introduction": updated_intro, "conclusion": updated_conclusion}

    @staticmethod
    def _insert_claim_tag(sentence: str, claim_id: str) -> str:
        if sentence and sentence[-1] in ".!?":
            return f"{sentence[:-1]} [{claim_id}]{sentence[-1]}"
        return f"{sentence} [{claim_id}]"

    def _build_embedding_document(
        self,
        article_id: str,
        submit_info: Dict[str, Any],
        source_context: Dict[str, Any],
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "meta": {
                "article_id": article_id,
                "canonical_source_mongo_id": source_context["source_doc_id"],
                "user_id": submit_info["user_id"],
                "type": "article_embedding",
                "embedding_model": settings["embedding_model"],
                "claim_count": len(self.claim_bank),
                "created_at": datetime.utcnow().isoformat(),
            },
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["text"],
                    "embedding": claim["embedding"],
                    "trust_label": claim["trust_label"],
                    "source_ids": claim["source_ids"],
                }
                for claim in self.claim_bank
            ],
        }

    def _persist_embedding_document(self, embedding_doc: Dict[str, Any], mongoio: MongoDBHandler):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return mongoio.write_document(embedding_doc)
            except Exception as exc:
                self.usage["retries"] += 1
                last_exc = exc
                if attempt == 3:
                    raise ArticleGenerationError(
                        f"Failed to persist embedding document: {exc}",
                        error_code="MONGO_ERROR",
                        failed_stage="persisting_document",
                        retryable=True,
                    ) from exc
                time.sleep(1)
        raise ArticleGenerationError(
            f"Failed to persist embedding document: {last_exc}",
            error_code="MONGO_ERROR",
            failed_stage="persisting_document",
            retryable=True,
        )

    @staticmethod
    def _strip_embeddings_from_filtered_retrieval(retrieval: Dict[str, Any]) -> Dict[str, Any]:
        sections = []
        for section in retrieval.get("sections", []):
            claims = [
                {k: v for k, v in claim.items() if k != "embedding"}
                for claim in section.get("claims", [])
            ]
            sections.append({**section, "claims": claims})
        return {"sections": sections}

    @staticmethod
    def _strip_embeddings_from_raw_retrieval(raw_retrieval: Dict[str, Any]) -> Dict[str, Any]:
        sections = []
        for section in raw_retrieval.get("sections", []):
            point_hits = []
            for point_hit in section.get("point_hits", []):
                hits = [
                    {k: v for k, v in hit.items() if k != "embedding"}
                    for hit in point_hit.get("hits", [])
                ]
                point_hits.append({**point_hit, "hits": hits})
            sections.append({**section, "point_hits": point_hits})
        return {"sections": sections}

    @staticmethod
    def _format_openai_error(exc: Exception) -> str:
        body = getattr(exc, "body", None)
        request_id = getattr(exc, "request_id", None)
        detail = body if body is not None else str(exc)
        if request_id:
            return f"OpenAI request failed: {detail} (request_id={request_id})"
        return f"OpenAI request failed: {detail}"

    def _merge_usage_bucket(self, bucket_name: str, usage):
        usage_dict = self._usage_to_dict(usage)
        for key, value in usage_dict.items():
            if key in self.usage[bucket_name]:
                self.usage[bucket_name][key] += value

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
    def _failure_dict(exc: ArticleGenerationError) -> Dict[str, Any]:
        return {
            "status": "FAILED",
            "message": exc.message,
            "error_code": exc.error_code,
            "failed_stage": exc.failed_stage,
            "retryable": exc.retryable,
            "exc_type": exc.exc_type,
            "exc_message": exc.message,
        }
