from src import deep_research as dr
from src.deep_research import FinalReportResponse, ProcessResult, ResearchImage


def test_process_result_coerces_object_learnings_without_dropping_metadata():
    result = ProcessResult.model_validate(
        {
            "learnings": [
                {
                    "summary": "Monte-Carlo evaluation found high variance across random initializations.",
                    "date": "Jul 2025",
                    "metrics": {"runs": 30, "p_value": 0.03},
                }
            ],
            "follow_up_questions": [{"question": "Which datasets were included?"}],
        }
    )

    assert len(result.learnings) == 1
    assert "Monte-Carlo evaluation found high variance" in result.learnings[0]
    assert '"p_value":0.03' in result.learnings[0]
    assert result.follow_up_questions == ['{"question":"Which datasets were included?"}']


def test_extract_image_candidates_keeps_surrounding_markdown_context():
    markdown = (
        "# Transformer-XL benchmark\n\n"
        "The following figure compares recurrence and segment-level attention on WikiText-103.\n\n"
        "![Transformer-XL architecture diagram](https://example.com/txl.png)\n\n"
        "The image is discussed next to perplexity and throughput results."
    )

    images = dr._extract_image_candidates(
        [{"url": "https://example.com/article", "markdown": markdown}],
        context_chars=120,
    )

    assert images == [
        {
            "image_url": "https://example.com/txl.png",
            "alt_text": "Transformer-XL architecture diagram",
            "source_url": "https://example.com/article",
            "markdown_image": "![Transformer-XL architecture diagram](https://example.com/txl.png)",
            "context": (
                "# Transformer-XL benchmark The following figure compares recurrence and "
                "segment-level attention on WikiText-103. ![Transformer-XL architecture "
                "diagram](https://example.com/txl.png) The image is discussed next to "
                "perplexity and throughput results."
            ),
        }
    ]


def test_final_report_appends_relevant_images_if_model_omits_them(monkeypatch):
    def fake_structured_completion(user_prompt, response_model):
        assert "<relevant_images>" in user_prompt
        assert "https://example.com/architecture.png" in user_prompt
        return FinalReportResponse(report_markdown="# Report\n\nUseful findings.")

    monkeypatch.setattr(dr, "_structured_completion", fake_structured_completion)

    report = dr.write_final_report(
        prompt="compare LSTM and Transformer architectures",
        learnings=["Transformer diagrams help explain the architecture."],
        visited_urls=["https://example.com/source"],
        relevant_images=[
            ResearchImage(
                image_url="https://example.com/architecture.png",
                alt_text="Architecture diagram",
                source_url="https://example.com/source",
                context="The diagram is adjacent to the architecture explanation.",
                relevance="Shows the architecture being compared.",
            )
        ],
    )

    assert "![Architecture diagram](https://example.com/architecture.png)" in report
    assert "Shows the architecture being compared." in report
    assert "- https://example.com/source" in report
