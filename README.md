Replicating the OOC experiments is very straightforward.
1) You can pip install the requirements from requirements.txt
2) Add the clinical notes dataset using the build function in _data (after adding the csv files from MIMIC-III and MIMIC-SBDH.
3) Choose the hyperparams of your run in setup.sh. Don't forget to add your openai key if you are using an OPENAI model.
4) run main.py
