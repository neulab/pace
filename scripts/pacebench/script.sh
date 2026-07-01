count=100

# python cli.py abs --count $count --B 300 --seed 42 --auto-tune --strict-budget > logs/abs_${count}.log
# python cli.py pair --count $count --B 300 --seed 42 --auto-tune --strict-budget --pin-task-a-selection > logs/pair_${count}.log

# python cli.py fit-abs --count $count --B 300 --seed 0 --auto-tune --strict-budget --dump-selections selections

python cli.py abs --count $count --B 300 --seed 42 --auto-tune --strict-budget
python cli.py pair --count $count --B 300 --seed 42 --auto-tune --strict-budget --pin-task-a-selection

# PROXYBENCH_NO_BOOTSTRAP=1 python cli.py abs  --count 100 \
#     --B 300 --seed 42 --auto-tune --strict-budget