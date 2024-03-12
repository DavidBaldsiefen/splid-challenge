
### Current general approach

- 1. Localize ID nodes with one model, then AD+IK with another
- 2. Add SS nodes
- 3. Run Classifier (assuming ID nodes are ID nodes for certain)

### TODOs & Ideas

- Train on whole dataset
- dynamic threshold that starts low, and then increases until n detections = n expected nodes?
- remove NS detections during EW-NK
- remove ID detections during SK (not sure what I meant here)
- if EW is NK, NS should be NK too - possibly add missing ID nodes?
- more regularization on classifier, less on localizer
- run two classifiers, one that just detects NK vs SK and then one that classifies between EK/CK/HK
- Overview fts as separate input
- Data Augumentation (e.g. noisy data)
- Scale on known max-min
- Make use of timestamp data in updated ds
- potentially improve clean consecutives by using approaches such as best-fit
- k-fold cross val

### Other interesting ideas for which I wont have time

- Autoregressive models
- Stateful LSTMs, Transformers
- Divide&Conquer style approach for localizers
- temporal embedding, temporal aggregation, TCN
- one-shot localizer (one pred that predicts whole object)


### Challenge reverse-engineering:

- number of objects: 162
- total number of nodes: (162*2)+396=720