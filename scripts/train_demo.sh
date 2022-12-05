mkdir -p /tmp/python
export TMPDIR="/tmp/python"

export DEBUG=1
cd /code
mkdir -p ./data
ln -s /train-data/training ./data/
ls data/training/
cp /train-data/reproduce.py ./
echo $PWD
/root/miniconda3/bin/python reproduce.py
cp -r lightning_logs /train-data/outputs
cd /train-data/
/root/miniconda3/bin/python format-ckpt.py
