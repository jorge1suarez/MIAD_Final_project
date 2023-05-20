sudo apt update; sudo apt upgrade -y
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda config --set auto_activate_base false

# echo 'export PATH=$PATH:/home/cyberithub/miniconda3/bin' | sudo tee -a /etc/profile
conda --version

conda create --name api -y
conda create --name etl -y


conda activate etl
conda install python=3.9 -y

conda install pandas numpy seaborn -y
pip install yfinance pandas_datareader ta
pip install psycopg2-binary sqlalchemy 
pip install google-cloud-logging google-cloud-storage


conda activate api
conda install python=3.10.11 -y

conda install pandas numpy scikit-learn -y
pip install yfinance pandas_datareader ta
pip install psycopg2-binary sqlalchemy scipy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-forecasting
pip install pytorch-lightning
pip install tensorboard
pip install optuna google-cloud-logging google-cloud-storage google-cloud-monitoring