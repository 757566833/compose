python3 -m venv venv
source ./venv/bin/activate 
python3 -m pip install --upgrade pip 
pip3 install -r requirements.txt    
NO_PROXY=192.168.246.22 HF_ENDPOINT=https://hf-mirror.com uvicorn app:app --host 0.0.0.0 --port 8000 --reload
