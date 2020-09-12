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

PID=1314
while [ -e /proc/$PID ]
do
    sleep .6
done

PID=684
while [ -e /proc/$PID ]
do
    sleep .6
done

parallel python train.py -c Experiments/configs/exp1.1.json
parallel python train.py -c Experiments/configs/exp1.2.json
parallel python train.py -c Experiments/configs/exp1.3.json
parallel python train.py -c Experiments/configs/exp2.1.json
parallel python train.py -c Experiments/configs/exp2.2.json
parallel python train.py -c Experiments/configs/exp2.3.json
parallel python train.py -c Experiments/configs/exp3.1.json
parallel python train.py -c Experiments/configs/exp3.2.json
parallel python train.py -c Experiments/configs/exp3.3.json
parallel python train.py -c Experiments/configs/exp3.4.json
parallel python train.py -c Experiments/configs/exp3.5.json
parallel python train.py -c Experiments/configs/exp3.6.json
parallel python train.py -c Experiments/configs/exp3.7.json
parallel python train.py -c Experiments/configs/exp3.8.json
parallel python train.py -c Experiments/configs/exp3.9.json
parallel python train.py -c Experiments/configs/exp3.10.json
parallel python train.py -c Experiments/configs/exp3.11.json
parallel python train.py -c Experiments/configs/exp3.12.json
wait
