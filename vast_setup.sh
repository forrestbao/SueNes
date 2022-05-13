# script to setup everything on a bare metal server

touch ~/.no_auto_tmux

echo "alias gputop='nvidia-smi --query-gpu=pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory --format=csv -l 1'" >> ~/.bashrc 

# first allow sync of data to start 
apt update
apt install vim rsync git  -y
git clone https://github.com/forrestbao/SueNes
mkdir data/

# then do the rest 
apt install wget rsync python3 python3-pip python3-venv python3-dev vim git zip htop screen -y 
apt update 
mkdir ~/bert_models
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip 
unzip uncased_L-12_H-768_A-12.zip -d bert_models/bert_base

python3 -m pip install pip setuptools requests --upgrade

# install Nvidia's own TF 1.15
python3 -m pip install nvidia-pyindex
python3 -m pip install nvidia-tensorflow[horovod]

# install stanza, spacy  and scipy 
# numpy should have been install with TF 
# install stanza will install pytorch

python3 -m pip install stanza
python3 -m pip install scipy
1python3 -m pip install spacy[cuda112]  # replace cuda112 with your cuda version
python3 -m spacy download en_core_web_sm

