python3 -m venv venv
source ./venv/bin/activate 
python3 -m pip install --upgrade pip 
#第一次安装
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install pip-tools -i https://mirrors.aliyun.com/pypi/simple
pip-compile requirements.txt --output-file requirements-locked.txt 
#已存在locked文件时使用
pip install -r requirements-locked.txt -i https://mirrors.aliyun.com/pypi/simple
NO_PROXY=192.168.246.22 HF_ENDPOINT=https://hf-mirror.com uvicorn app:app --host 0.0.0.0 --port 8000 --reload
