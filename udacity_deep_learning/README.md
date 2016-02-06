# Create virtual environment
mkdir ./tensorflow
virtualenv ./tensorflow
source ~/tensorflow/bin/activate

# Install pip requirements
pip install -r requirements.txt

# Download and generate data files
python process_data.py

