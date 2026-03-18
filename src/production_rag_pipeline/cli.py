import argparse
import json

from . import ask_with_search, save_prompt_to_file, search, search_and_read, search_bing, search_ddg
from .config import configure


def _detect_lang(query):
    return "ru" if any("\u0400" <= char <= "\u04ff" for char in query) else "en"


def build_parser():
    parser = argparse.ArgumentParser(prog="production-rag-pipeline")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--mode",
        choices=["llm", "read", "debug", "search", "bing", "ddg"],
        default="llm",
        help="Execution mode",
    )
    parser.add_argument("--lang", choices=["en", "ru", "zh"], help="Force query language")
    parser.add_argument("--config", help="Path to YAML config file")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Введите поисковый запрос:\n> ").strip()
        if not query:
            parser.error("query cannot be empty")

    lang = args.lang or _detect_lang(query)
    configure(path=args.config)

    if args.mode == "bing":
        result = search_bing(query, lang=lang, debug=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "ddg":
        result = search_ddg(query, lang=lang, debug=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "debug":
        result = search(query, num=20, lang=lang, debug=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "read":
        print(search_and_read(query, lang=lang))
        return

    if args.mode == "search":
        result = search(query, num=20, lang=lang)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prompt = ask_with_search(query, lang=lang)
    save_prompt_to_file(query, prompt)
    print(prompt)


if __name__ == "__main__":
    main()
