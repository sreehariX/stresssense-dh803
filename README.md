# StressSense

DH803 Wearable Health Technologies course project. We ask whether wrist recordings can distinguish a study's stress task from baseline or rest for a new participant, and how much performance changes in a second study.

Each prediction uses 30 seconds of blood volume pulse, skin conductance, wrist temperature and movement recorded by an Empatica E4. The output is a class label and a model score. Researchers assigned the class according to the task being performed. The score expresses how much the recording resembles the study's positive class.

## Run the experiment

Use Python 3.11 or 3.12. These commands run from the root of this repository. Creating a virtual environment is optional but recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl -fL -o data/WESAD.zip https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
curl -fL -o data/physionet-1.0.1.zip https://physionet.org/content/wearable-device-dataset/get-zip/1.0.1/
python -m stresssense.prepare
python -m stresssense.evaluate
python -m stresssense.figures
python -m unittest discover -s tests
```

On Windows PowerShell, activate the environment with `./.venv/Scripts/Activate.ps1`, create the data directory with `New-Item -ItemType Directory -Force data` and use `curl.exe` for the two downloads. Both archives stay zipped. Allow about 2.4 GB for the archives and a few minutes for feature extraction. Use only the original, trusted WESAD archive for its legacy pickle files. The loaders also accept explicit paths via `--wesad` and `--physionet`.

The preparation step saves feature tables and an audit to `run/`. Evaluation writes a machine readable summary, window predictions, participant scores and the fitted WESAD model to the same directory. Logistic regression scored 89.2% balanced accuracy on held out WESAD participants and 68.9% on PhysioNet. The [report](REPORT.md) describes the results and limitations. The [aggregate output](results/) gives the exact numbers behind it.

## Replay an evaluated recording

```bash
python -m stresssense.replay physionet S01 data/physionet-1.0.1.zip --output run/replay.html
```

Open `run/replay.html` in a browser. Move through the windows to compare the traces, study label and model prediction. To replay a WESAD participant, use `wesad S2 data/WESAD.zip` for the three positional arguments. The generated page embeds the recording locally and stays outside Git.

## How the experiment works

1. Read the synchronized WESAD wrist signals and keep source labels 1 and 2. Match the PhysioNet signals and protocol tags by timestamp.
2. Ignore transitions and place nonoverlapping 30 second windows at least five seconds inside each labeled period. Extract pulse and pulse interval statistics, skin conductance and temperature summaries, and movement measures.
3. Hold out every WESAD participant once. Fit median imputation and any scaling on the training participants in each fold. Compare fixed logistic regression and random forest models, an always rest baseline and a model that sees only movement.
4. Fit the same models on all of WESAD and apply them to PhysioNet without refitting or calibrating on PhysioNet. Report the average of each metric across people and a participant bootstrap interval.

The repository includes code and aggregate results. Original recordings, window features, fitted models, and replays are excluded. WESAD is available for scientific, noncommercial use with attribution. PhysioNet v1.0.1 uses the Open Data Commons Attribution License. The [references](REFERENCES.md) link the original datasets, papers and methods.
