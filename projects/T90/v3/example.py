from __future__ import annotations

from pprint import pprint

from interface import recommend_t90_v3


def main() -> None:
    example_request = {
        "objective_hint": "future_window_warning",
        "dcs_window": "placeholder",
        "ph_history": None,
    }
    result = recommend_t90_v3(example_request)
    pprint(result)


if __name__ == "__main__":
    main()
