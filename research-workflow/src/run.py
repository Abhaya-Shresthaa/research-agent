import asyncio
import os

from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

from src.ai.providers import get_model_id
from src.deep_research import deep_research, write_final_answer, write_final_report
from src.feedback import generate_feedback


def _log(*args, **kwargs):
    print(*args, **kwargs)


def _ask(question: str) -> str:
    try:
        return input(question)
    except EOFError:
        return ""


async def run():
    _log("Using model: ", get_model_id())

    # Get initial query
    initial_query = _ask("What would you like to research? ")

    # Breadth and depth
    breadth_raw = _ask("Enter research breadth (recommended 2-10, default 4): ")
    breadth = int(breadth_raw) if breadth_raw.strip().isdigit() else 4

    depth_raw = _ask("Enter research depth (recommended 1-5, default 2): ")
    depth = int(depth_raw) if depth_raw.strip().isdigit() else 2

    is_report = (
        _ask("Do you want to generate a long report or a specific answer? (report/answer, default report): ")
        != "answer"
    )

    combined_query = initial_query

    if is_report:
        _log("Creating research plan...")

        follow_up_questions = await generate_feedback(query=initial_query)
        _log(
            "\nTo better understand your research needs, please answer these follow-up questions:"
        )

        answers: list[str] = []
        for question in follow_up_questions:
            answer = _ask(f"\n{question}\nYour answer: ")
            answers.append(answer)

        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)
        )
        combined_query = (
            f"Initial Query: {initial_query}\n"
            f"Follow-up Questions and Answers:\n{qa_pairs}"
        )

    _log("\nStarting research...\n")

    result = await deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
    )

    _log("\nResearch complete.")
    _log("Writing final output...")

    if is_report:
        report = write_final_report(
            prompt=combined_query,
            learnings=result.learnings,
            visited_urls=result.visited_urls,
            relevant_images=result.relevant_images,
        )
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report)
        # print(f"\n\nFinal Report:\n\n{report}")
        print("\nReport has been saved to report.md")
    else:
        answer = write_final_answer(
            prompt=combined_query,
            learnings=result.learnings,
        )
        with open("answer.md", "w", encoding="utf-8") as f:
            f.write(answer)
        print(f"\n\nFinal Answer:\n\n{answer}")
        print("\nAnswer has been saved to answer.md")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
