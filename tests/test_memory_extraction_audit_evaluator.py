import json

import pytest

from memory_engine.memory.extraction_auditor import (
    ExtractionAuditIssue,
    ExtractionAuditIssueType,
    FakeMemoryExtractionAuditor,
    MemoryExtractionAudit,
)
from memory_engine.memory.extractor import AssertionMemoryFact, MemoryFactSet
from memory_engine.models import MemoryType
from evaluation.evaluators.memory_extraction_audit_evaluator import (
    run_evaluation,
)


class StaticExtractor:
    async def extract(self, user_message):
        return MemoryFactSet(
            facts=(
                AssertionMemoryFact(
                    kind="assertion",
                    content="The user prefers remote work.",
                    memory_type=MemoryType.PREFERENCE,
                    evidence="I prefer remote work",
                    confidence=0.99,
                ),
            )
        )


@pytest.mark.asyncio
async def test_audit_evaluator_records_recovery_without_regression(tmp_path):
    message = "I prefer remote work and I decided to save for a house"
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "LLMMemoryFactExtractor",
        "description": "Unit audit fixture",
        "evaluation_contract": {"acceptance_gates": {}},
        "cases": [
            {
                "id": "compound",
                "tags": ["compound"],
                "user_message": message,
                "expected": {
                    "facts": [
                        {
                            "kind": "assertion",
                            "memory_type": "preference",
                            "content_assertions": {
                                "must_include": ["remote work"]
                            },
                            "accepted_evidence": ["I prefer remote work"],
                        },
                        {
                            "kind": "assertion",
                            "memory_type": "decision",
                            "content_assertions": {
                                "must_include": ["save", "house"]
                            },
                            "accepted_evidence": [
                                "I decided to save for a house"
                            ],
                        },
                    ]
                },
            }
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    final_facts = MemoryFactSet(
        facts=(
            (await StaticExtractor().extract(message)).facts[0],
            AssertionMemoryFact(
                kind="assertion",
                content="The user decided to save for a house.",
                memory_type=MemoryType.DECISION,
                evidence="I decided to save for a house",
                confidence=0.99,
            ),
        )
    )
    auditor = FakeMemoryExtractionAuditor(
        default=MemoryExtractionAudit(
            approved=False,
            issues=(
                ExtractionAuditIssue(
                    issue_type=ExtractionAuditIssueType.MISSING_FACT,
                    description="A durable decision was missing.",
                ),
            ),
            final_facts=final_facts,
        )
    )

    report = await run_evaluation(
        dataset_path,
        report_path,
        extractor=StaticExtractor(),
        auditor=auditor,
    )

    assert report["baseline_summary"]["fact_recall"] == 0.5
    assert report["audited_summary"]["fact_recall"] == 1.0
    assert report["audit_summary"]["recovery_rate"] == 1.0
    assert report["audit_summary"]["regression_rate"] == 0.0
