import os

from dotenv import load_dotenv
from google import genai

from code_index_manager import load_or_build_code_index
from workflow import run_workflow


load_dotenv()


def main() -> None:
    print("AI Software Engineer Agent — Production Workflow")
    print("Enter a software engineering task.")
    print()

    task = input("Task: ").strip()

    if not task:
        print("Error: Please enter a task.")
        return

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY is missing.")
        return

    try:
        print("\n[Main] Creating Gemini client...")
        client = genai.Client(api_key=api_key)

        print("[Main] Loading/building code index...")
        code_store = load_or_build_code_index(
            client=client,
        )

        print(f"[Main] Code index ready: {len(code_store)} chunks.")

        result = run_workflow(
            task,
            client=client,
            code_store=code_store,
            embedding_client=client,
        )

        print("\nWorkflow result")
        print("----------------")
        print(f"Status: {result.status}")
        print(f"Stages: {' -> '.join(result.stages)}")

        if result.error:
            print(f"Error: {result.error}")

        if result.output:
            print("\nOutput")
            print("------")
            print(result.output)

    except KeyboardInterrupt:
        print("\n[Main] Workflow stopped by user.")

    except Exception as exc:
        message = str(exc).lower()

        if "429" in message or "resource_exhausted" in message:
            print("\nAgent error: Gemini API quota is exhausted.")
            print(
                "The application started successfully, but the required "
                "Gemini operation could not complete."
            )
            print("Please check your Gemini API quota or billing and try again later.")
        else:
            print(f"\nAgent error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
