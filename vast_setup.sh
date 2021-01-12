# script to setup everything on a bare metal server

touch ~/.no_auto_tmux

echo "alias gputop='nvidia-smi --query-gpu=pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory --format=csv -l 1'" >> ~/.bashrc 


# first allow sync of data to start 
sudo apt install rsync git 
git clone https://github.com/forrestbao/anti-rouge
mkdir anti-rouge/data

# then do the rest 
sudo apt install wget rsync python3 python3-pip python3-venv vim git zip htop
mkdir ~/bert_models
mkdir ~/bert_models/bert_base
cd ~/bert_models/bert_base
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip 
unzip uncased_L-12_H-768_A-12.zip 
cd ~

python3 -m pip install pip --upgrade
python3 -m venv --system-site-packages tf1 
source tf1/bin/activate 
python3 -m pip install tensorflow-gpu==1.15 scipy


