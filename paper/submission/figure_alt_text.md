# Figure Alt Text

## Figure 1: Method pipeline

A left-to-right five-stage workflow. Post-tonal conditions, including pitch-class
sets or interval vectors, row forms, rhythm, and gesture, are serialized by
condition encoders. A causal Transformer generates candidate score events. A
constraint-guided decoder evaluates multiple candidates, then exports a
MusicXML score and an analysis report.

## Figure 2: Full-run automatic effects

Ten small paired plots in two rows. The top row compares one candidate with four
guided candidates from a shared generator for pc-set coverage, interval-vector
distance, row-order accuracy, rhythmic-profile distance, and gesture
consistency. Every top-row metric moves in the preferred direction. The bottom
row compares a full-prefix single-candidate model with four separately trained
condition-removal models. Removing each condition worsens its associated
metric. Values are descriptive results for 2,000 synthetic test conditions per
experiment.

## Figure 3: Representative MusicXML score

One rendered page from proposed-model example project2_20. Four labeled staves
contain a sparse four-measure post-tonal fragment in 4/4. Notes span low and high
registers with sustained values, rests, and several ties. The requested pc set
is 0, 1, 2, 5, and 6; the rhythmic profile is sustained and the gesture is
registral expansion.
