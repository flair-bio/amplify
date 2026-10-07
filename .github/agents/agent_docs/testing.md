### Running Tests
- Full suite: `make test` (runs `pytest tests -n auto` in parallel with pytest-xdist)
- While iterating, run only the relevant files serially, e.g. `pytest tests/pretrain/test_collator.py`; parallel start-up costs more than it saves on small runs
- Run the full suite once before reporting done
