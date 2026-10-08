import os
from dotenv import load_dotenv
from google import genai
from models import TaskResponse


load_dotenv()


class LLMServiceError(Exception):
    """Raised when the LLM service cannot complete a request."""


SYSTEM_INSTRUCTION = """
You are the planning foundation of an AI Software Engineer Agent.

Analyze the user's software development task.
Return a concise summary, an ordered implementation plan,
relevant assumptions, and potential technical or security risks.

Do not claim that code has been inspected, changed, or tested.
You do not have access to the user's repository or execution tools.
State reasonable assumptions explicitly when details are missing.
"""


def analyze_task(task: str, client=None) -> TaskResponse:
    """Analyze a development task and return validated structured data."""

    if not task or not task.strip():
        raise ValueError("Task must not be empty.")

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if client is None:
        if not api_key:
            raise LLMServiceError(
                "GEMINI_API_KEY is missing. Add it to your local .env file."
            )

        client = genai.Client(api_key=api_key)

    try:
        interaction = client.interactions.create(
            model=model,
            input=f"{SYSTEM_INSTRUCTION}\n\nUser task:\n{task}",
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": TaskResponse.model_json_schema(),
                }
            ],
        )

        if not interaction.output_text:
            raise ValueError("Gemini returned an empty response.")

        return TaskResponse.model_validate_json(
            interaction.output_text
        )

    except Exception as exc:
        raise LLMServiceError(
            f"Gemini request or response validation failed: {exc}"
        ) from exc
