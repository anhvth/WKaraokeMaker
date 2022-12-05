docker run -it --gpus "device=0"  -v `pwd`:/training/ --name zac2022-training -p 9777:9777 zac2022:v1 /bin/bash 
