from pydantic import BaseModel

from src.ai.providers import get_client, get_model_id
from src.prompt import system_prompt


class FeedbackResponse(BaseModel):
    questions: list[str]


async def generate_feedback(
    query: str,
    num_questions: int = 3,
) -> list[str]:
    """Generate follow-up questions to clarify the research direction."""
    client = get_client()
    model_id = get_model_id()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Given the following query from the user, ask some follow up questions "
                    f"to clarify the research direction. Return a maximum of {num_questions} "
                    f"questions, but feel free to return less if the original query is clear. "
                    f"Respond with JSON: {{\"questions\": [\"...\"]}}: "
                    f"<query>{query}</query>"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    parsed = FeedbackResponse.model_validate_json(content)
    return parsed.questions[:num_questions]
