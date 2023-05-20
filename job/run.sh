cd /home/freddy_12_120/MIAD_Final_project/etl/
conda activate etl
python3 etl_first.py
conda deactivate

cd /home/freddy_12_120/MIAD_Final_project/training/
conda activate api
python3 validation.py
python3 train_cpu.py

cd /home/freddy_12_120/MIAD_Final_project/predictions/
python3 predictions.py
conda deactivate
