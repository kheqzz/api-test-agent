
from app.llm import get_llm


def main() -> None:
    llm = get_llm()
    response = llm.invoke(
        "Balas dengan tepat: LLM gateway berhasil."
    )
    print(response.content)


if __name__ == "__main__":
    main()
