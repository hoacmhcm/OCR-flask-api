# OCR-flask-api
conda config --append channels conda-forge

conda activate OCR-flask-api
pip install -r requirements.txt

gunicorn -w 4 'app:app'

ngrok http --domain=magical-robin-thankfully.ngrok-free.app 80