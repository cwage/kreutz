#!/bin/bash
# Lint and the unit tests, inside the project container (what CI runs,
# minus the import check). The real detector test is the July 2025
# benchmark, scripts/benchmark.sh, run by hand when the detector changes.
cd "$(dirname "$0")/.."
docker compose run --rm -T -e HOME=/tmp -e PYTHONUSERBASE=/tmp/py -v "$PWD/tests:/srv/tests" -v "$PWD/requirements-dev.txt:/srv/requirements-dev.txt" --entrypoint sh kreutz -c "pip install -q --user --no-warn-script-location -r /srv/requirements-dev.txt 2>&1 | grep -v notice; cd /srv && python -m pyflakes kreutz tests && python -m pytest -q -p no:cacheprovider tests"
