"""Entry point for `python -m hayward`, equivalent to the `hayward` command."""

from hayward.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
