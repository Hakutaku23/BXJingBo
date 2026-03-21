from __future__ import annotations

import json

from interface import recommend_t90_controls


def main() -> None:
    result = recommend_t90_controls({"use_example_data": True})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
