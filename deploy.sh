pip install -r requirements.txt
nohup uvicorn app:app --host 0.0.0.0 --port 8000 &
python -m streamlit run frontend.py --server.port 8001 --server.address 0.0.0.0