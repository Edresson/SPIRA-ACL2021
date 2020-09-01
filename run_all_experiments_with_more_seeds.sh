#python train.py -c Experiments/configs/exp3.3.json &
#python train.py -c Experiments/configs/exp3.4.json

# My gpu memory support two processes
max_children=2
function parallel {
  # Credits: Arnaldo Candido Junior
  local time1=$(date +"%H:%M:%S")
  local time2=""

  echo "starting $1 ($time1)..."
  "$@" && time2=$(date +"%H:%M:%S") && echo "finishing $2 ($time1 -- $time2)..." &

  local my_pid=$$
  local children=$(ps -eo ppid | grep -w $my_pid | wc -w)
  children=$((children-1))
  if [[ $children -ge $max_children ]]; then
    wait -n
  fi
}

PID=13561
while [ -e /proc/$PID ]
do
    sleep .6
done


seed_s=43
seed_e=43
for i in $(seq $seed_s 1 $seed_e)
do
  # parallel python train.py -c Experiments/configs/exp1.1.json -s $i
  # parallel python train.py -c Experiments/configs/exp1.2.json -s $i
  # parallel python train.py -c Experiments/configs/exp1.3.json -s $i
  # parallel python train.py -c Experiments/configs/exp2.1.json -s $i
  # parallel python train.py -c Experiments/configs/exp2.2.json -s $i
  # parallel python train.py -c Experiments/configs/exp2.3.json -s $i
  parallel python train.py -c Experiments/configs/exp3.1.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.2.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.3.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.4.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.5.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.6.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.7.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.8.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.9.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.10.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.11.json -s $i
  # parallel python train.py -c Experiments/configs/exp3.12.json -s $i
done

wait
