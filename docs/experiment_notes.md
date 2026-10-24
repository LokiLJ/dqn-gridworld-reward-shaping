# Experiment notes

This project originated from an **MMA 606 course assignment**, with the implementation adapted from the assignment notebook. It explores how map design, movement costs, network capacity, and reward shaping affect a Deep Q-Network (DQN) navigating a gridworld. An optional SVM predicts actions from its replay buffer.

## Evidence and the central discrepancy

Three source versions were supplied:

| Source | Evidence retained |
| --- | --- |
| `Assignment.docx` | Iteration narrative, four images, and SVM results |
| Initially supplied assignment notebook | A 3×3 bounded starter environment, four actions, one hidden layer of width 128; no swamp, dash, beacon, or SVM implementation |
| Final supplied notebook | The extended 6×6 environment, eight actions, beacon shaping, tapered DQN, SVM code, training log, and trajectory |

**The report records a successful run with return 1013, while the final notebook's saved run finishes at −33.** Its checkpoint labeled 145 is −78, rather than the report image's 1013. The [saved training log](../assets/notebook/training_log.txt) and [51-step trajectory](../assets/notebook/trajectory.txt) preserve this difference; the trajectory's rewards sum to −33 and show repeated cycling without completing the task. These are separate historical artifacts, not a new training run or a reproduction of the report's success.

## Verified environment

The final code uses this 6×6 map:

```text
S R F S - C
S F - - - F
- F - F - -
- - A - S -
- S S F - -
- S S - - F
```

`A` is the agent, `C` and `R` are crystals, `F` is fire, `S` is swamp, and `-` is empty. The top-left cell is swamp, but the bottom-right cell remains **fire**, despite the report describing an earlier replacement with swamp. The final code does not preserve every narrated map variant.

Movement and observation wrap around both dimensions. The observation is a flattened **3×3 local window × six one-hot cell types = 54 features**, not the full map. Local observations can omit distant crystals and other state information. Direction labels follow the notebook's coordinate convention: Left/Right modify the first array coordinate; Up/Down modify the second.

Normal actions move one cell with base reward −1 and an additional −2 for swamp. Dash actions traverse up to three cells with base reward −2 and −1 for each swamp traversed. Fire causes −1000 and terminates immediately, including during a dash. Crystals are collected only on landing, adding +5; collecting the last adds +1000 and terminates.

The map stores the agent by overwriting its landing cell. Departure then writes an empty cell, so previously occupied swamp disappears. Intermediate swamp crossed during a dash remains. This behavior is preserved, rather than silently replaced with persistent terrain.

### Beacon reward

After updating the map, each action adds:

```text
beacon = 2 × max(0, 3 − d)
d = minimum toroidal Manhattan distance to any remaining crystal
```

The bonus is zero if no crystals remain. It is applied even on a terminal fire step. Because collected crystals have already been overwritten, they no longer attract the beacon.

The report's “3×3 area” description is approximate: distance two along an axis also earns a bonus, so the implemented region is a Manhattan-distance neighborhood rather than a square. This is a direct proximity bonus, not a difference between successive state potentials. The notebook establishes no guarantee that shaping preserves the original optimal policy; reward can accrue near a crystal without reaching it.

## Verified DQN configuration and evaluation

| Setting | Final source value |
| --- | --- |
| Network | **54 → 256 → 128 → 8**, ReLU after each hidden layer |
| Discount / optimizer / loss | 0.99 / Adam at learning rate 0.1 / mean squared error |
| Replay capacity / batch size | 100 transitions / 32 |
| Training limit | 200 episodes, up to 50 actions each |
| Target updates | Copy online weights every 20 completed episodes |
| Exploration | Epsilon starts at 1.0; exponential decay toward 0.05 with scale 200 |
| Evaluation | Greedy actions; periodic single rollout, final 20 rollouts |
| Early stopping | Periodic evaluation return at least 1000 |

The online and target networks are initialized separately, without an initial weight synchronization. No random seeds are fixed. Episode truncation at the training horizon is not stored as a terminal transition. These choices are retained as part of the historical implementation.

The loop increments its completed-episode counter and then prints `ep + 1`: label 145 corresponds to 144 completed episodes in this implementation. Evaluation uses `rolloutStep <= 50`, allowing **51** actions, while training ends at 50. This explains why an otherwise uneventful evaluation can return −51.

The final 20 evaluations reset the same deterministic map and use the same greedy network. They are repeated rollouts of one fitted policy, not 20 independent training seeds or a generalization test. Nonconsecutive execution counts also leave earlier interactive execution history unknown.

## Reported iteration history

“Final mean” below preserves the source's label “mean reward after training.” Earlier variants are reported observations; their exact code and checkpoints are not retained.

| Stage | Change | Historical result | Evidence |
| --- | --- | --- | --- |
| Initial 5×5 | Wraparound, swamp, dash | Final mean −83 | Report prose |
| Deeper 5×5 network | Adds hidden linear layer and ReLU | One reward 998; final mean −87 | Report prose |
| Wider 5×5 network | Hidden width 256 | Final mean 1000 | Report prose |
| Initial 6×6 | More terrain constraints | Final mean −102 | Report prose |
| Bottom-right fire → swamp | Changes access | Final mean −52 | Report prose |
| Tapered network | Changes layer widths | Around −52 | Report prose |
| Top-left fire → swamp | Further map change | Final mean **−99** | [Original log](../assets/report/without_beacon.png) |
| Initial beacon | Adds proximity reward | Final mean **−51** | [Original log](../assets/report/beacon_initial.png) |
| Beacon multiplier 1.5 | Stronger shaping | Final mean **55** | Report prose |
| Beacon multiplier 2 | Stronger shaping | Label **145** and final mean **1013** | [Original log](../assets/report/beacon_multiplier_2.png) |
| Final notebook artifact | Saved extended implementation | Label 145: **−78**; final mean **−33** | [Notebook log](../assets/notebook/training_log.txt) |

The first report screenshot has some −51 checkpoints but ends with mean −99; it does not establish stable performance at −51. The second screenshot is the initial beacon trial, not multiplier 1.5. The report's proposed one-crystal explanation for return 55 has no accompanying trajectory.

Map, architecture, and rewards changed across trials, so this is not a controlled ablation. A shaped return of 1013 and an unshaped negative return also measure different reward functions. The successful report image supports a historical observation, not an optimality proof, reliable convergence, or generalization to new maps. The unsuccessful final saved run is a material limit on those interpretations.

## SVM action prediction

The code fits `SVC(kernel='rbf')` to states and actions from the rolling 100-transition training buffer, with other SVM settings left at library defaults. It then predicts on **the same states used for fitting**. There is no held-out split, cross-validation, feature scaling, or hyperparameter search. The original source imported a splitting function but never used it; that unused import was removed from the public copy.

The [saved classification report](../assets/notebook/svm_classification_report.txt) matches the report's [confusion matrix](../assets/report/svm_confusion_matrix.png): **64 correct predictions out of 100**, macro F1 **0.41**, weighted F1 **0.59**. This is **training accuracy**, not test accuracy.

| Action | Support | Correct | Recall |
| --- | ---: | ---: | ---: |
| Left | 10 | 4 | 0.40 |
| Right | 5 | 0 | 0.00 |
| Up | 30 | 27 | 0.90 |
| Down | 17 | 10 | 0.59 |
| Dash-L | 8 | 5 | 0.62 |
| Dash-R | 22 | 18 | 0.82 |
| Dash-U | 5 | 0 | 0.00 |
| Dash-D | 3 | 0 | 0.00 |

Up and Dash-R supply 52% of labels. Right, Dash-U, and Dash-D receive no predictions. Uniform random prediction has nominal accuracy 12.5%; always predicting Up gives 30% on these labels. Neither comparison demonstrates generalization.

The buffer contains epsilon-greedy training actions, including exploration, rather than labels exclusively produced by a converged greedy policy. Repeated local observations can also omit distinctions relevant to the action. The result therefore diagnoses limited action coverage and fit on this buffer; imbalance alone is not established as the cause, and successful SVM-controlled navigation was not evaluated in the supplied materials.

## Reproduction and provenance

No new DQN or SVM training was used to produce the historical results above. The successful run's weights, random seed, full trajectory, and exact environment snapshot were not supplied. Package versions are not locked. Re-running the notebook may produce different results.

A stronger follow-up would compare fixed maps across multiple training seeds, measure crystal completion separately from shaped return, and test action prediction on excluded episodes. Those would be new experiments, not corrections to the historical scores.

SHA-256 fingerprints identify the unmodified source files; both notebook versions originally shared a filename:

```text
Final MMA_606_Assignment_3_Notebook.ipynb
6ecc2e7cab6ad36b6a7502d08cd6e0f56ecb444259876ea26553f91567c3d7b5

Initially supplied starter notebook
211c123bd16ab5e9ee30cae66ef33a9a3712bdf14042243068021698040cd3e7

Assignment.docx
739e3118e36872f9c9493e7847c48dda01752fadeee34c034ee03a2b38fec7e8
```

Course provenance is retained without inventing an upstream author. Private document-sharing links and local machine paths are omitted.

### Publication cleanup and validation

The public notebook retains the active map, environment transitions, beacon function, network architecture, hyperparameters, training and evaluation loops, and SVM fitting procedure. Unused imports, assignment directions, numbered editing reminders, execution counts, outputs, and Colab metadata were removed. Historical text outputs and report images are stored separately. Both SVM displays now specify all eight labels, so a new replay buffer with missing action classes can still be displayed correctly; the fitted model and predictions are unchanged.

Eleven environment regression tests cover wraparound observations, movement, swamp costs, intermediate fire, crystal pickup, beacon behavior, reset, terrain clearing, and the saved trajectory. Independently replaying all 51 saved actions reproduces every saved reward and observation, with return −33 and one crystal remaining. A network forward-pass and single optimizer-update smoke check also passes, with matching results before and after cleanup. These checks validate code behavior without reproducing training or claiming a new learning result.

Publication checks used Python 3.11.4, PyTorch 2.5.1+cu121 on CPU, NumPy 2.3.5, scikit-learn 1.7.2, Matplotlib 3.10.7, Seaborn 0.13.2, and nbformat 5.9.2. This is the inspection environment, not a reconstruction of the original training environment. The new map illustration is drawn from the active map literal; the report figures are unchanged extracts. Credential-pattern scans found no hardcoded tokens or passwords in the files selected for publication.
