
### TODOs

- TODO: find out if we can directly train for precision/challenge metrics. Maybe at least precision?
- TODO: there is some kind of bug in the prediction_models where callbacks get maintained even when changed
- TODO: Make use of timestamp data in updated ds
- Maybe predict for whole timeseries at once? As shitty baseline?
- Try localizing nodes first, then classifying them. The latter could be autoregressive, as nodes depend on the previous nodes. However, the training should use the real past outputs maybe?
- I definetely need to preprocess more values -> Longitude has weird value range, negative vals but also vals over 180?
- combine own localizer with heursitics classification
- use class weight in localizer to penalize FN more than FP -> should improve recall (strong class imbalance)
- Try a 2-step classifier: 1) classify only the ID/AD/IK nodes 2) identify the single propulsion type per direction per object
- should probably really implement k fold crossval, at least for fast algorithms (like the classifier)


### Challenge reverse-engineering:

- number of objects: 162
- total number of nodes: (162*2)+396=720