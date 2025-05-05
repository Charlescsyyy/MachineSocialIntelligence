# Empowerment Lander

Codebase built on work from https://github.com/yuqingd/ave/tree/master/empowerment_lander
To set up required packages, use `environment.yml` with conda or `requirements.txt` with pip.
If using conda, activate the environment using `conda activate lander`. Then install OpenAI Baselines
using `pip install -e baselines`.

For Human solo: `python run_scripts/human_solo.py `.

Human and Empowerment: `python run_scripts/human_emp.py --empowerment`.

Train RLGI to obtain `policies/goal_rl.pkl`:`python run_scripts/train_goal_inference.py --steps 500000 --save policies/goal_rl.pkl`.

Human+Empowerment+RLGI: `python run_scripts/human_emp.py --empowerment --goal_rl policies/goal_rl.pkl`



