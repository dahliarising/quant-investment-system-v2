#!/bin/bash

# 개선된 프로젝트 구조 생성
mkdir -p config
mkdir -p engine
mkdir -p data/{providers,bundles}
mkdir -p features/{factors,pipeline}
mkdir -p models
mkdir -p validation
mkdir -p portfolio
mkdir -p services
mkdir -p ui/{components,utils}
mkdir -p utils
mkdir -p tests
mkdir -p notebooks
mkdir -p scripts
mkdir -p data_cache
mkdir -p logs

# __init__.py 파일 생성
find . -type d -not -path "./venv*" -not -path "./.git*" -not -path "./data_cache*" -not -path "./logs*" -exec touch {}/__init__.py \;

echo "프로젝트 구조 생성 완료!"
