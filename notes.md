
### TODOs

- TODO: find out if we can directly train for precision/challenge metrics. Maybe at least precision?
- TODO: Make sure labelencoder gets saved
- TODO: there is some kind of bug in the prediction_models where callbacks get maintained even when changed
- TODO: Make use of timestamp data in updated ds
- Maybe predict for whole timeseries at once? As shitty baseline?
- Try localizing nodes first, then classifying them. The latter could be autoregressive, as nodes depend on the previous nodes. However, the training should use the real past outputs maybe?
- I definetely need to preprocess more values -> Longitude has weird value range, negative vals but also vals over 180?
- idea: make ds gen always use fast compute, then just slice away unnecessary values! not ideal though, as it will cut away too much on one side
- combine own localizer with heursitics classification
- use class weight in localizer to penalize FN more than FP -> should improve recall (strong class imbalance)
- there is no node for ending station keeping... can specific nodes only be followed by specific other nodes?