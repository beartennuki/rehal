import re
from typing import Any, Dict, List


class ArticleQualityChecker:
    CLAIM_TAG_RE = re.compile(r"\[(C\d{4})\]")
    SOURCE_TAG_RE = re.compile(r"\[\^(S\d{3})\]")

    def __init__(
        self,
        claim_id_to_claim: Dict[str, Dict[str, Any]],
        sid_to_url: Dict[str, str],
        claim_bank: List[Dict[str, Any]],
    ):
        self.claim_id_to_claim = claim_id_to_claim
        self.sid_to_url = sid_to_url
        self.claim_bank = claim_bank

    def run_quality_checks(
        self,
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        article_with_claim_ids: str,
        article_with_sources: str,
        untagged_findings: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        coverage_report = self.build_coverage_report(
            retrieval,
            body_sections,
            intro_conclusion,
            article_with_claim_ids,
            untagged_findings=untagged_findings,
        )
        unresolved_claim_ids = [
            claim_id for claim_id in coverage_report["cited_claim_ids"]
            if claim_id not in self.claim_id_to_claim
        ]
        empty_sections = [section["subtopic"] for section in body_sections if not section["body"].strip()]
        missing_claim_sections = [
            section["subtopic"] for section in retrieval["sections"] if not section["claims"]
        ]
        missing_source_urls = [
            source_id for source_id in self.extract_source_ids(article_with_sources)
            if source_id not in self.sid_to_url or not self.sid_to_url[source_id]
        ]
        uncited_sections = [
            section["subtopic"]
            for section in coverage_report["sections"]
            if section["cited_claim_count"] == 0
        ]
        word_count = len(article_with_sources.split())
        if untagged_findings is None:
            untagged_findings = self.locate_untagged_factual_sentences(
                body_sections,
                intro_conclusion,
            )
        unresolved_findings = self.locate_unresolved_claim_ids(
            article_with_claim_ids, unresolved_claim_ids, body_sections, intro_conclusion
        )
        uncited_findings = [
            {
                "check": "cited_claims_per_section",
                "location": f"section:{subtopic}",
                "subtopic": subtopic,
                "sentence": "",
                "suggested_action": "regenerate_section",
            }
            for subtopic in uncited_sections
        ]
        missing_source_findings = [
            {
                "check": "missing_source_urls",
                "location": self.locate_source_tag(article_with_sources, source_id),
                "source_id": source_id,
                "sentence": "",
                "suggested_action": "rewrite_sentence_with_claim_tags",
            }
            for source_id in missing_source_urls
        ]

        checks = {
            "has_sections": bool(body_sections),
            "has_citations": (
                coverage_report["total_cited_claims"]
                / max(coverage_report["total_retrieved_claims"], 1)
            ) >= 0.90,
            "untagged_factual_sentences": len(untagged_findings) == 0,
            "unresolved_claim_ids": not unresolved_claim_ids,
            "missing_claim_sections": not missing_claim_sections,
            "missing_source_urls": not missing_source_urls,
            "non_empty_sections": not empty_sections,
            "cited_claims_per_section": not uncited_sections,
            "minimum_word_count": word_count >= 300,
        }

        failures = [name for name, passed in checks.items() if not passed]
        findings = untagged_findings + unresolved_findings + uncited_findings + missing_source_findings
        repairable_failures = set(failures) - {"missing_claim_sections", "has_sections"}
        repairable = bool(repairable_failures)

        return {
            "passed": not failures,
            "checks": checks,
            "failures": failures,
            "repairable": repairable,
            "coverage_report": coverage_report,
            "unresolved_claim_ids": unresolved_claim_ids,
            "missing_source_urls": missing_source_urls,
            "empty_sections": empty_sections,
            "uncited_sections": uncited_sections,
            "word_count": word_count,
            "findings": findings,
            "repair_instructions": self.summarize_repair_targets(findings, failures, body_sections),
        }

    def build_coverage_report(
        self,
        retrieval: Dict[str, Any],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
        final_article: str,
        untagged_findings: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        cited_claim_ids = set(self.extract_claim_ids(final_article))
        retrieved_claim_ids = set()
        section_reports = []

        for section in retrieval["sections"]:
            section_claim_ids = [claim["claim_id"] for claim in section["claims"]]
            retrieved_claim_ids.update(section_claim_ids)
            body = next(
                (item["body"] for item in body_sections if item["subtopic"] == section["subtopic"]), ""
            )
            section_cited = set(self.extract_claim_ids(body))
            section_reports.append({
                "subtopic": section["subtopic"],
                "retrieved_claim_count": len(section_claim_ids),
                "cited_claim_count": len(section_cited),
                "unused_retrieved_claim_ids": sorted(set(section_claim_ids) - section_cited),
                "cited_claim_ids": sorted(section_cited),
            })

        effective_untagged_findings = (
            untagged_findings
            if untagged_findings is not None
            else self.locate_untagged_factual_sentences(body_sections, intro_conclusion)
        )
        untagged_factual_sentences = [
            finding.get("sentence", "")
            for finding in effective_untagged_findings
            if finding.get("sentence")
        ]
        return {
            "total_claim_bank_size": len(self.claim_bank),
            "total_retrieved_claims": len(retrieved_claim_ids),
            "total_cited_claims": len(cited_claim_ids),
            "unused_retrieved_claims": sorted(retrieved_claim_ids - cited_claim_ids),
            "cited_claim_ids": sorted(cited_claim_ids),
            "untagged_factual_sentence_count": len(untagged_factual_sentences),
            "untagged_factual_sentences": untagged_factual_sentences,
            "sections": section_reports,
        }

    def build_exception_quality(self, message: str) -> Dict[str, Any]:
        return {
            "passed": False,
            "checks": {},
            "failures": ["pipeline_error"],
            "repairable": False,
            "coverage_report": {
                "total_claim_bank_size": len(self.claim_bank),
                "total_retrieved_claims": 0,
                "total_cited_claims": 0,
                "unused_retrieved_claims": [],
                "cited_claim_ids": [],
                "untagged_factual_sentence_count": 0,
                "untagged_factual_sentences": [],
                "sections": [],
            },
            "unresolved_claim_ids": [],
            "missing_source_urls": [],
            "empty_sections": [],
            "uncited_sections": [],
            "word_count": 0,
            "findings": [],
            "repair_instructions": {},
            "message": message,
        }

    @staticmethod
    def build_quality_summary(quality: Dict[str, Any]) -> Dict[str, Any]:
        quality = quality or {}
        coverage = quality.get("coverage_report", {})
        return {
            "passed": quality.get("passed", False),
            "failures": quality.get("failures", []),
            "total_cited_claims": coverage.get("total_cited_claims", 0),
            "untagged_factual_sentence_count": coverage.get("untagged_factual_sentence_count", 0),
            "unresolved_claim_count": len(quality.get("unresolved_claim_ids", [])),
        }

    def locate_untagged_factual_sentences(
        self,
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        findings = []
        for sentence in self.flag_untagged_factual_sentences(intro_conclusion.get("introduction", "")):
            findings.append({
                "check": "untagged_factual_sentences",
                "location": "introduction",
                "sentence": sentence,
                "suggested_action": "rewrite_sentence_with_claim_tags",
            })
        for section in body_sections:
            for sentence in self.flag_untagged_factual_sentences(section.get("body", "")):
                findings.append({
                    "check": "untagged_factual_sentences",
                    "location": f"section:{section['subtopic']}",
                    "subtopic": section["subtopic"],
                    "sentence": sentence,
                    "suggested_action": "regenerate_section",
                })
        for sentence in self.flag_untagged_factual_sentences(intro_conclusion.get("conclusion", "")):
            findings.append({
                "check": "untagged_factual_sentences",
                "location": "conclusion",
                "sentence": sentence,
                "suggested_action": "rewrite_sentence_with_claim_tags",
            })
        return findings

    def locate_unresolved_claim_ids(
        self,
        article_with_claim_ids: str,
        unresolved_claim_ids: List[str],
        body_sections: List[Dict[str, Any]],
        intro_conclusion: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        findings = []
        for claim_id in unresolved_claim_ids:
            location = "article"
            if claim_id in intro_conclusion.get("introduction", ""):
                location = "introduction"
            elif claim_id in intro_conclusion.get("conclusion", ""):
                location = "conclusion"
            else:
                for section in body_sections:
                    if claim_id in section.get("body", ""):
                        location = f"section:{section['subtopic']}"
                        break
            findings.append({
                "check": "unresolved_claim_ids",
                "location": location,
                "sentence": self.extract_sentence_containing(article_with_claim_ids, claim_id),
                "claim_id": claim_id,
                "subtopic": location.split("section:", 1)[1] if location.startswith("section:") else None,
                "suggested_action": "rewrite_sentence_with_claim_tags",
            })
        return findings

    def locate_source_tag(self, article_with_sources: str, source_id: str) -> str:
        token = f"[^{source_id}]"
        for sentence in re.split(r"(?<=[.!?])\s+", article_with_sources):
            if token in sentence:
                return self.infer_location_from_rendered_article(sentence)
        return "article"

    @staticmethod
    def infer_location_from_rendered_article(sentence: str) -> str:
        return "article"

    def summarize_repair_targets(
        self,
        findings: List[Dict[str, Any]],
        failures: List[str],
        body_sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        section_targets = sorted({
            finding.get("subtopic")
            for finding in findings
            if finding.get("subtopic")
        })
        locations = {finding.get("location") for finding in findings}
        regenerate_all_sections = False
        if "missing_claim_sections" in failures or not section_targets and any(
            failure in failures for failure in ["minimum_word_count", "has_sections", "non_empty_sections"]
        ):
            regenerate_all_sections = True
        if not section_targets and "untagged_factual_sentences" in failures and "introduction" not in locations and "conclusion" not in locations:
            regenerate_all_sections = True
        if not section_targets and "unresolved_claim_ids" in failures and all(
            location in {"introduction", "conclusion", "article"} for location in locations if location
        ):
            regenerate_all_sections = True

        return {
            "section_targets": section_targets,
            "repair_intro": "introduction" in locations or regenerate_all_sections,
            "repair_conclusion": "conclusion" in locations or regenerate_all_sections,
            "regenerate_all_sections": regenerate_all_sections,
        }

    @staticmethod
    def extract_source_ids(article_with_sources: str) -> List[str]:
        return sorted(set(re.findall(r"\[\^(S\d{3})\]", article_with_sources)))

    def flag_untagged_factual_sentences(self, article: str) -> List[str]:
        sentences = self.sentence_split(article)
        return [
            sentence for sentence in sentences
            if self.looks_factual(sentence) and not self.has_inline_citation(sentence)
        ]

    def has_inline_citation(self, sentence: str) -> bool:
        return bool(self.CLAIM_TAG_RE.search(sentence) or self.SOURCE_TAG_RE.search(sentence))

    @staticmethod
    def looks_factual(sentence: str) -> bool:
        sentence = sentence.strip()
        if len(sentence.split()) < 8:
            return False
        lowered = sentence.lower()
        if lowered.startswith(("overall,", "overall ", "in summary", "to conclude", "in practice", "for readers")):
            return False
        factual_markers = [
            r"\b\d{3,4}\b",
            r"\b\d+(\.\d+)?%",
            r"\baccording to\b",
            r"\bdata from\b",
            r"\bfigures from\b",
            r"\bstatistics from\b",
            r"\bresearchers\b",
            r"\bstudy\b",
            r"\bstudies\b",
            r"\bevidence (shows|suggests|indicates)\b",
            r"\bhistorically\b",
            r"\bhas shown\b",
            r"\bhas found\b",
            r"\bwas found\b",
            r"\bwere found\b",
            r"\bhas reported\b",
            r"\bwas reported\b",
            r"\bwere reported\b",
            r"\bresearch\b",
            r"\bdata\b",
            r"\bmeasured\b",
            r"\breported\b",
            r"\bestimate[sd]?\b",
            r"\bforecast[sd]?\b",
        ]
        return any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in factual_markers)

    def extract_claim_ids(self, text: str) -> List[str]:
        return sorted(set(self.CLAIM_TAG_RE.findall(text)))

    @staticmethod
    def extract_sentence_containing(text: str, token: str) -> str:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if token in sentence:
                return sentence.strip()
        return ""

    @staticmethod
    def sentence_split(text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        return re.split(r"(?<=[.!?])\s+", normalized)
