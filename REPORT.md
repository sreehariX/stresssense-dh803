# StressSense final report

DH803 Wearable Health Technologies

## Research question

Can 30 seconds of recordings from a wristband distinguish a stress task from baseline or rest for a person absent from training? Does the same model work when the recordings come from a second study? Earlier work tested stress models across studies and found that performance depended on the study, the device and the task [5]. We addressed the question with the same wrist device in two studies so the comparison could focus on how a model transfers between their recordings.

Logistic regression reached 89.2% balanced accuracy on held out WESAD participants and 68.9% on PhysioNet. The transfer result and its error counts describe the limits of this model under a second set of controlled tasks.

## Data and labels

| Data used | Participants | Baseline or rest windows | Stress windows |
| --- | ---: | ---: | ---: |
| WESAD | 15 | 578 | 322 |
| PhysioNet v1.0.1 | 34 | 1,357 | 276 |

Both studies recorded blood volume pulse, skin conductance, wrist temperature and acceleration with the Empatica E4. We used WESAD's synchronized wrist signals and retained its baseline and Trier Social Stress Test labels. In PhysioNet we read the UTC timestamps from the four streams and aligned them to the authors’ event markers. The source notebook defines a separate mapping for the two protocol versions. We treated initial baseline and scheduled rests as the negative class and labeled the documented tasks as the positive class. The shorter speech and subtraction blocks often yielded no full analysis window. Mental arithmetic accounted for 208 of 276 PhysioNet stress windows [1–4].

We excluded S02 because the source documents duplicated signals and f07 because the protective dock covered two sensors. For f14 we used the tagged part of the split recording. Its untagged baseline was omitted. Exercise, amusement and unassigned time were outside the analysis. Participant and window totals and archive checksums are recorded in the [data audit](results/data_audit.json).

## Signal processing and evaluation

The pipeline uses consecutive 30 second windows, leaving a five second guard at each end of a labeled block. It uses the original sampling rate for each sensor. Acceleration is converted to units of g. The features are:

| Signal | Features |
| --- | --- |
| Blood volume pulse | Median interval converted to pulse rate and interquartile range of valid pulse intervals |
| Skin conductance and wrist temperature | Mean, standard deviation, slope and 90th minus 10th percentile |
| Acceleration | Spread of vector magnitude, mean variation across axes and average change per second |

The pulse extractor uses a 0.6 to 4 Hz bandpass filter and peak detection. An interval is accepted in the 0.33 to 1.50 second range. The window needs enough accepted intervals to cover at least 18 seconds. This feature is a short window pulse estimate. Wrist motion can distort the optical measurement [7, 12]. The two pulse features were missing in 53 WESAD windows and 18 PhysioNet windows. No other feature was missing in this run. The model fills missing values with training medians.

We used logistic regression with class weights and fixed regularization (C = 1) as the primary model. Comparison models included a random forest with 200 trees and a minimum of five samples per leaf, a logistic regression using acceleration alone, and a classifier that always predicts rest. For the internal test, we held out one WESAD participant at a time and fitted the full processing pipeline on the other 14. For the external test, we fitted on all 15 WESAD participants, kept the decision cutoff at 0.5 and applied the model to PhysioNet without refitting. We followed the same procedure when we omitted one sensor at a time [6, 9, 10].

Balanced accuracy gives equal weight to stress recall and rest recall. We calculated each metric within a participant and then averaged across participants. The 95% percentile intervals resample participant scores 5,000 times. They describe sample variation conditional on these evaluations and do not include the uncertainty of retraining on a new source study.

## Results

| Model | WESAD held out people | PhysioNet transfer |
| --- | ---: | ---: |
| Always predicts rest | 50.0% | 50.0% |
| Movement only | 84.8% | 58.8% |
| Logistic regression | 89.2% | 68.9% |
| Random forest | 88.5% | 62.8% |

Table values are mean participant balanced accuracy. The intervals for the primary model were 84.2% to 93.4% on WESAD and 64.1% to 73.2% on PhysioNet. Stress recall changed from 89.6% to 73.3%, and rest recall changed from 88.8% to 64.4%. Stress F1, which combines stress precision and recall, fell from 86.2% to 44.6%. On PhysioNet the primary model missed 77 of 276 stress windows and called 440 of 1,357 rest windows stress. Those two counts pool windows across people.

![Balanced accuracy with participant bootstrap intervals](results/model_comparison.png)

| Logistic regression inputs | WESAD | PhysioNet |
| --- | ---: | ---: |
| All four signals | 89.2% | 68.9% |
| Without pulse | 86.1% | 61.5% |
| Without skin conductance | 87.4% | 66.6% |
| Without wrist temperature | 88.1% | 65.7% |
| Without acceleration | 84.6% | 67.0% |

Removing pulse caused the largest drop in PhysioNet for this set of models. Removing acceleration caused a larger drop within WESAD than it did in PhysioNet. The movement only result shows that task behavior carried useful information in the WESAD protocol. These comparisons cannot tell us which physiological process caused the model output.

### How the comparison condition affected the result

The primary model scored 65.1% balanced accuracy on PhysioNet's first protocol and 72.6% on its second. These were separate cohorts with changes in their tasks and rest schedules. The comparison cannot isolate a single cause. Among the 33 people with a labeled initial baseline, balanced accuracy was 58.4% when we kept the stress windows and used only that initial baseline as the negative class. This is a sensitivity analysis with the same fitted model and cutoff. It shows how strongly the result depends on which intervals count as rest.

## What the result supports

The model learned patterns that helped it separate study conditions for unseen people. Performance declined by roughly 20 percentage points when the study changed. PhysioNet's lower F1 and frequent false alarms matter if the intended use is to flag short, relatively infrequent stress tasks. An average score from one protocol would give a misleading picture of performance in a new setting.

These are task labels from two controlled studies. People can experience different levels of stress while doing the same task, and movement, speech and recovery can differ across task blocks. Age, device placement, protocol and study effects were not separated. The evaluation did not test ordinary activities or use perceived stress as the prediction target. An independent evaluation with movement matched activities or data collected in everyday life would address a different and harder question.

The [reproduction instructions](README.md) cover data preparation, model fitting and the offline recording replay. The [exact output](results/summary.json) contains all models, confusion matrices and uncertainty intervals. The [bibliography](REFERENCES.md) links the dataset documentation and research sources.
