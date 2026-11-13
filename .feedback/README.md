# Feedback task starting environment

Upstream: https://github.com/fox-it/dissect.hypervisor
Upstream commit: `9736c31a0761d96eb81d22a3ba675c07b67f1d26`
Prepared on macOS with the Python version recorded in the environment plan. No external database, queue, or cloud account is required.

The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC UV_NO_CONFIG=1

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= tests
```

Scope note: Complete offline upstream test scope specified by argv.

Observed baseline (2026-11-13): `57 passed, 1 warning in 10.90s`
Whole-command wall time: 11.240 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
