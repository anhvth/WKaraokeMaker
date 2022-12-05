#unzip ../training_data.zip
if [[ $1=="demo" ]];then
    docker run -it --gpus "device=0"  -v $(pwd)/:/train-data/ --name reproduce-zac2022 zac2022:v1 /bin/bash /train-data/train.sh
else
    docker run -it --gpus "device=0,1,2,3,4,5"  -v $(pwd)/:/train-data/ zac2022:v1 /bin/bash /train-data/train.sh
fi
