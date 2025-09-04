# Formation: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-formation/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-formation)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

## About the project

## Setup
1. Make sure you have python version 3.11 or at least 3.10  
Check by running `python --version`. If you have older version please update.

2. Create virtual environment and activate it
```shell
python -m venv .venv && source .venv/bin/activate
```
3. Upgrade pip
```shell
python -m pip install --upgrade pip
```
4. Install runtime dependencies with `pip install -r requirements.txt` command

## Usage
To run the program after setup, use the `python main.py` command in the torchRL folder. The output is printed to the standard output.

## Running tests
```shell
python -m unittest discover -s test
```

## Work distribution:
Attila: Gif generation, Presentation, Docker implementation, Unit tests.
Árpád: TorchRL implementation skeleton, and then improvements, Wandb.