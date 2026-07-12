import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from firecrawl import FirecrawlApp
from pydantic import AliasChoices, BaseModel, Field, field_validator

from src.ai.providers import get_client, get_model_id, trim_prompt
from src.prompt import system_prompt

# ── Helpers / Logging ──────────────────────────────────────────────────────

def _log(*args, **kwargs):
    pass


# ── Dataclasses / Pydantic models ──────────────────────────────────────────

@dataclass
class ResearchProgress:
    current_depth: int = 0
    total_depth: int = 0
    current_breadth: int = 0
    total_breadth: int = 0
    current_query: Optional[str] = None
    total_queries: int = 0
    completed_queries: int = 0


@dataclass
class ResearchResult:
    learnings: list[str] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)
    relevant_images: list["ResearchImage"] = field(default_factory=list)


class SerpQuery(BaseModel):
    query: str
    research_goal: str


class SerpQueriesResponse(BaseModel):
    queries: list[SerpQuery]


class ResearchImage(BaseModel):
    image_url: str = Field(default="", validation_alias=AliasChoices("image_url", "imageUrl", "url"))
    alt_text: str = Field(default="", validation_alias=AliasChoices("alt_text", "altText", "alt"))
    source_url: str = Field(default="", validation_alias=AliasChoices("source_url", "sourceUrl", "source"))
    context: str = Field(default="")
    relevance: str = Field(default="")

    @field_validator("image_url", "alt_text", "source_url", "context", "relevance", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)


class ProcessResult(BaseModel):
    learnings: list[str]
    follow_up_questions: list[str]
    relevant_images: list[ResearchImage] = Field(default_factory=list)

    @field_validator("learnings", "follow_up_questions", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [_learning_to_string(item) for item in value if item is not None]

    @field_validator("relevant_images", mode="before")
    @classmethod
    def _coerce_image_list(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [{"image_url": item} if isinstance(item, str) else item for item in value if item is not None]


class FinalAnswerResponse(BaseModel):
    exact_answer: str


class FinalReportResponse(BaseModel):
    report_markdown: str


# ── Firecrawl client ───────────────────────────────────────────────────────

_concurrency_limit = int(os.environ.get("FIRECRAWL_CONCURRENCY", "2"))

_firecrawl_kwargs = {"api_key": os.environ.get("FIRECRAWL_KEY", "")}
if os.environ.get("FIRECRAWL_BASE_URL"):
    _firecrawl_kwargs["api_url"] = os.environ["FIRECRAWL_BASE_URL"]

_firecrawl = FirecrawlApp(**_firecrawl_kwargs)


_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\((<[^>]+>|[^\s)]+)(?:\s+[\"'][^\"']*[\"'])?\)"
)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _learning_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        summary = value.get("summary") or value.get("learning") or value.get("text")
        rest = {k: v for k, v in value.items() if k not in {"summary", "learning", "text"}}
        if summary and rest:
            return f"{summary} | metadata: {_compact_json(rest)}"
        if summary:
            return str(summary)
    return _compact_json(value)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_image_candidates(
    result_data: list[dict],
    *,
    max_images_per_page: int = 8,
    max_total_images: int = 40,
    context_chars: int = 700,
) -> list[dict]:
    candidates: list[dict] = []
    seen_urls: set[str] = set()

    for item in result_data:
        markdown = item.get("markdown") or ""
        source_url = item.get("url") or ""
        page_count = 0
        for match in _MARKDOWN_IMAGE_RE.finditer(markdown):
            raw_image_url = match.group(2).strip()
            image_url = raw_image_url[1:-1] if raw_image_url.startswith("<") and raw_image_url.endswith(">") else raw_image_url
            if not image_url or image_url.startswith("data:") or image_url in seen_urls:
                continue

            start = max(0, match.start() - context_chars)
            end = min(len(markdown), match.end() + context_chars)
            context = _normalize_whitespace(markdown[start:end])

            candidates.append(
                {
                    "image_url": image_url,
                    "alt_text": _normalize_whitespace(match.group(1)),
                    "source_url": source_url,
                    "markdown_image": match.group(0),
                    "context": context,
                }
            )
            seen_urls.add(image_url)
            page_count += 1

            if page_count >= max_images_per_page or len(candidates) >= max_total_images:
                break
        if len(candidates) >= max_total_images:
            break

    return candidates


def _dedupe_images(images: list[ResearchImage]) -> list[ResearchImage]:
    deduped: list[ResearchImage] = []
    seen_urls: set[str] = set()
    for image in images:
        if not image.image_url or image.image_url in seen_urls:
            continue
        deduped.append(image)
        seen_urls.add(image.image_url)
    return deduped


# ── LLM helpers ────────────────────────────────────────────────────────────

def _structured_completion(user_prompt: str, response_model: type[BaseModel]) -> BaseModel:
    """Call the LLM with JSON-mode output and validate with a Pydantic model."""
    client = get_client()
    model_id = get_model_id()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return response_model.model_validate_json(content)


# ── SERP query generation ─────────────────────────────────────────────────

def _generate_serp_queries(
    query: str,
    num_queries: int = 3,
    learnings: Optional[list[str]] = None,
) -> list[SerpQuery]:
    learnings_hint = ""
    if learnings:
        learnings_hint = (
            f"\n\nHere are some learnings from previous research, use them to "
            f"generate more specific queries: {' '.join(learnings)}"
        )

    prompt = (
        f"Given the following prompt from the user, generate a list of SERP queries "
        f"to research the topic. Return a maximum of {num_queries} queries, but feel "
        f"free to return less if the original prompt is clear. Make sure each query is "
        f"unique and not similar to each other: <prompt>{query}</prompt>{learnings_hint}\n\n"
        f"Respond with JSON: {{\"queries\": [{{\"query\": \"...\", \"research_goal\": \"...\"}}]}}"
    )

    res = _structured_completion(prompt, SerpQueriesResponse)
    _log(f"Created {len(res.queries)} queries", [q.query for q in res.queries])
    return res.queries[:num_queries]


# ── Process SERP result ──────────────────────────────────────────────────

def _process_serp_result(
    query: str,
    result_data: list[dict],
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> ProcessResult:
    contents = [
        trim_prompt(item.get("markdown") or "", 25_000)
        for item in result_data
        if item.get("markdown")
    ]
    _log(f"Ran {query}, found {len(contents)} contents")

    contents_xml = "\n".join(
        f"<content>\n{c}\n</content>" for c in contents
    )
    image_candidates = _extract_image_candidates(result_data)
    images_xml = "\n".join(
        (
            "<image_candidate>\n"
            f"<image_url>{img['image_url']}</image_url>\n"
            f"<alt_text>{img['alt_text']}</alt_text>\n"
            f"<source_url>{img['source_url']}</source_url>\n"
            f"<context>{img['context']}</context>\n"
            "</image_candidate>"
        )
        for img in image_candidates
    )
    prompt = trim_prompt(
        f"Given the following contents from a SERP search for the query "
        f"<query>{query}</query>, generate a list of learnings from the contents. "
        f"Return a maximum of {num_learnings} learnings, but feel free to return less "
        f"if the contents are clear. Make sure each learning is unique and not similar "
        f"to each other. The learnings should be comprehensive, detailed, and information-dense. "
        f"Each learning should capture a meaningful insight, concept, data point, or relationship "
        f"from the content — not just a single fact but the surrounding context and significance "
        f"where possible. Make sure to include any entities like people, places, companies, "
        f"products, technologies, frameworks, architectures, etc in the learnings, as well as "
        f"any exact metrics, numbers, dates, comparisons, mechanisms, causal relationships, "
        f"or step-by-step processes. The learnings will be used to write a comprehensive "
        f"final report, so prioritize depth and completeness over brevity.\n\n"
        f"<contents>{contents_xml}</contents>\n\n"
        f"Optionally, some image candidates extracted from the pages are included below. "
        f"Only select an image if its surrounding context clearly shows it is a genuinely "
        f"important diagram, chart, architecture figure, or key illustration that directly "
        f"supports understanding of the research topic. Be very selective — skip logos, "
        f"avatars, ads, tracking pixels, generic social cards, decorative images, stock photos, "
        f"and any image whose value is not clearly demonstrated by its context.\n\n"
        f"<image_candidates>{images_xml}</image_candidates>\n\n"
        f"Respond with JSON: {{\"learnings\": [\"...\"], "
        f"\"follow_up_questions\": [\"...\"], "
        f"\"relevant_images\": [{{\"image_url\": \"...\", \"alt_text\": \"...\", "
        f"\"source_url\": \"...\", \"context\": \"...\", \"relevance\": \"...\"}}]}}"
    )

    res = _structured_completion(prompt, ProcessResult)
    _log(f"Created {len(res.learnings)} learnings", res.learnings)
    if res.relevant_images:
        _log(f"Selected {len(res.relevant_images)} relevant images", [img.image_url for img in res.relevant_images])
    return res


# ── Final answer / report ─────────────────────────────────────────────────

def write_final_answer(prompt: str, learnings: list[str]) -> str:
    learnings_xml = "\n".join(
        f"<learning>\n{l}\n</learning>" for l in learnings
    )
    user_prompt = trim_prompt(
        f"Given the following prompt from the user, write a final answer on the topic "
        f"using the learnings from research. Follow the format specified in the prompt. "
        f"Do not yap or babble or include any other text than the answer besides the "
        f"format specified in the prompt. Keep the answer as concise as possible - "
        f"usually it should be just a few words or maximum a sentence. Try to follow "
        f"the format specified in the prompt (for example, if the prompt is using Latex, "
        f"the answer should be in Latex. If the prompt gives multiple answer choices, "
        f"the answer should be one of the choices).\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research on the topic that you can use to "
        f"help answer the prompt:\n\n<learnings>\n{learnings_xml}\n</learnings>\n\n"
        f"Respond with JSON: {{\"exact_answer\": \"...\"}}"
    )
    res = _structured_completion(user_prompt, FinalAnswerResponse)
    return res.exact_answer


def write_final_report(
    prompt: str,
    learnings: list[str],
    visited_urls: list[str],
    relevant_images: Optional[list[ResearchImage]] = None,
) -> str:
    learnings_xml = "\n".join(
        f"<learning>\n{l}\n</learning>" for l in learnings
    )
    relevant_images = relevant_images or []
    images_xml = "\n".join(
        (
            "<image>\n"
            f"<image_url>{img.image_url}</image_url>\n"
            f"<alt_text>{img.alt_text}</alt_text>\n"
            f"<source_url>{img.source_url}</source_url>\n"
            f"<context>{img.context}</context>\n"
            f"<relevance>{img.relevance}</relevance>\n"
            "</image>"
        )
        for img in relevant_images
        if img.image_url
    )
    user_prompt = trim_prompt(
        f"Given the following prompt from the user, write a final report on the topic "
        f"using the learnings from research. Make it as detailed as possible, aim for "
        f"3 or more pages, include ALL the learnings from research. Cover key concepts, "
        f"architectures, mechanisms, metrics, comparisons, relationships, and any "
        f"step-by-step processes found in the learnings:\n\n"
        f"<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from previous research:\n\n"
        f"<learnings>\n{learnings_xml}\n</learnings>\n\n"
        f"{f'<relevant_images>\n{images_xml}\n</relevant_images>\n\n' if images_xml else ''}"
        f"If relevant images are provided above, embed the most important ones in the "
        f"report using markdown syntax: ![alt text](image_url). Only include images that "
        f"illustrate key concepts, data, architectures, or entities directly related to "
        f"the report content — skip decorative or marginal ones.\n\n"
        f"Respond with JSON: {{\"report_markdown\": \"...\"}}"
    )
    res = _structured_completion(user_prompt, FinalReportResponse)
    urls_section = "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in visited_urls)
    report_markdown = res.report_markdown

    included_image_urls = {img.image_url for img in relevant_images if img.image_url and img.image_url in report_markdown}
    missing_images = [img for img in relevant_images if img.image_url and img.image_url not in included_image_urls]
    if missing_images:
        images_section = "\n\n## Relevant Images\n\n" + "\n\n".join(
            (
                f"![{img.alt_text or 'Relevant research image'}]({img.image_url})\n\n"
                f"*{img.relevance or img.context or 'Selected as relevant to the research topic.'}*"
                + (f"\n\nSource: {img.source_url}" if img.source_url else "")
            )
            for img in missing_images[:10]
        )
        report_markdown += images_section

    return report_markdown + urls_section


# ── Core recursive research loop ───────────────────────────────────────────

async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    learnings: Optional[list[str]] = None,
    visited_urls: Optional[list[str]] = None,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
) -> ResearchResult:
    learnings = learnings or []
    visited_urls = visited_urls or []

    progress = ResearchProgress(
        current_depth=depth,
        total_depth=depth,
        current_breadth=breadth,
        total_breadth=breadth,
    )

    def report_progress(**kwargs):
        for k, v in kwargs.items():
            setattr(progress, k, v)
        if on_progress:
            on_progress(progress)

    serp_queries = _generate_serp_queries(
        query=query,
        num_queries=breadth,
        learnings=learnings,
    )

    report_progress(
        total_queries=len(serp_queries),
        current_query=serp_queries[0].query if serp_queries else None,
    )

    semaphore = asyncio.Semaphore(_concurrency_limit)

    async def _run_query(serp_query: SerpQuery) -> ResearchResult:
        async with semaphore:
            try:
                result = _firecrawl.search(
                    serp_query.query,
                    timeout=15000,
                    limit=5,
                    scrape_options={"formats": ["markdown"]},
                )

                # ── Normalize firecrawl v2 SearchData → list of dicts ──
                # SearchData.web contains SearchResultWeb or Document objects.
                items: list[dict] = []
                web_results = getattr(result, "web", None) or []
                for item in web_results:
                    # URL: Document stores it in metadata.url / metadata.source_url;
                    # SearchResultWeb stores it directly in .url
                    url = ""
                    if isinstance(item, dict):
                        url = item.get("url", "")
                        markdown = item.get("markdown", "")
                    else:
                        url = getattr(item, "url", "") or ""
                        if not url:
                            md = getattr(item, "metadata", None)
                            if md is not None:
                                url = getattr(md, "url", "") or getattr(md, "source_url", "") or ""
                                if not url and isinstance(md, dict):
                                    url = md.get("url", "") or md.get("sourceURL", "") or ""
                        markdown = getattr(item, "markdown", "") or ""
                    items.append({"url": url, "markdown": markdown})

                new_urls = [d["url"] for d in items if d["url"]]

                new_breadth = (breadth + 1) // 2
                new_depth = depth - 1

                # Process SERP result
                processed = _process_serp_result(
                    query=serp_query.query,
                    result_data=items,
                    num_follow_up_questions=new_breadth,
                )
                all_learnings = learnings + processed.learnings
                all_urls = visited_urls + new_urls
                all_images = _dedupe_images(processed.relevant_images)

                if new_depth > 0:
                    _log(f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}")
                    report_progress(
                        current_depth=new_depth,
                        current_breadth=new_breadth,
                        completed_queries=progress.completed_queries + 1,
                        current_query=serp_query.query,
                    )

                    follow_ups = "\n".join(f"\n{q}" for q in processed.follow_up_questions)
                    next_query = (
                        f"Previous research goal: {serp_query.research_goal}\n"
                        f"Follow-up research directions: {follow_ups}"
                    ).strip()

                    deeper_result = await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        on_progress=on_progress,
                    )
                    deeper_result.relevant_images = _dedupe_images(all_images + deeper_result.relevant_images)
                    return deeper_result
                else:
                    report_progress(
                        current_depth=0,
                        completed_queries=progress.completed_queries + 1,
                        current_query=serp_query.query,
                    )
                    return ResearchResult(
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        relevant_images=all_images,
                    )

            except Exception as e:
                if "Timeout" in str(e):
                    _log(f"Timeout error running query: {serp_query.query}: ", e)
                else:
                    _log(f"Error running query: {serp_query.query}: ", e)
                return ResearchResult()

    results = await asyncio.gather(*[_run_query(q) for q in serp_queries])

    all_learnings = list({l for r in results for l in r.learnings})
    all_urls = list({u for r in results for u in r.visited_urls})
    all_images = _dedupe_images([img for r in results for img in r.relevant_images])
    return ResearchResult(
        learnings=all_learnings,
        visited_urls=all_urls,
        relevant_images=all_images,
    )
