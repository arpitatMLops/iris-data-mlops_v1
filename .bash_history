cd /home/sagemaker-user
git clone https://github.com/arpitatMLops/iris-data-mlops.git
git config --global user.name "Arpita"
git config --user.email "arpitat4@gmail.com"
git config --global user.email "arpitat4@gmail.com"
ls src/preprocessing
cd home/sagemaker-user
cd iris-data-mlops
ls
cd iris-data-mlops/src/preprocessing
cd iris-data-mlops/src
cd iris-data-mlops/src/preprocessing
ls
python src/preprocessing/preprocessing.py
cd iris-data-mlops/src/preprocessing
ls
python iris-data-mlops/src/preprocessing/preprocessing.py
python preprocessing.py
# from /home/sagemaker-user/iris-data-mlops
python - <<PY
from src.preprocessing.preprocessing import main
main(output_dir="./processed_output_local")
print("Done. Check ./processed_output_local/processed.csv")
PY

ls -l processed_output_local
head -n 5 processed_output_local/processed.csv
python <<PY
from src.preprocessing.preprocessing import load_data
load_data(output_dir = "./processed_output")
print(done)
PY

cd iris-data-mlops
