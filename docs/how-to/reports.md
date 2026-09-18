# Reports

`mesoscopy report` writes a self-contained HTML report for a pipeline output, picking the report from the
filename suffix. The report is written next to the input, or under `-o`:

```bash
mesoscopy report /path/to/recording_preprocessed.h5
mesoscopy report /path/to/recording_registered.h5
mesoscopy report /path/to/recording_smoothed_regions_event-cueonset_perievent.csv -o /path/to/reports
```

Reports internet connection to render but are otherwise self-contained.

## Preprocessing report

For a `*_preprocessed.h5` written by `preprocess`. Shows the session metadata, the automated QA checks
(histogram separation, timestamp consistency and jumps, noise, SNR, bleaching), the timestamp series and
its jumps, the per-frame statistics used for channel separation and the resulting frame assignment, the
mean ∆F/F of each channel, and the haemodynamics-corrected ∆F/F with its projections and an example frame.

## Registration report

For a `*_registered.h5` written by `register landmarks`. Shows the recording landmarks against the template
landmarks before and after the affine transform, and a registered frame with the template landmarks overlaid.

## Peri-event report

For a `*_perievent.csv` written by `process peri-event`. Shows each region's response around the event as the
mean ± 95% CI across trials or as individual trials, filtered by trial type, one region at a time and as a grid
of every region. The region is chosen from a dropdown, by clicking the Allen CCF top view, or by clicking a
panel in the grid.

When the `*_metrics.csv` and `*_metrics-session.csv` from `process metrics` sit next to the input, the
report adds a metrics section: onset, peak and offset markers on the traces with the across-trial spread of
each, per-trial box plots for the selected region, the atlas coloured by a session metric, and the session
table. Peri-event files written without the trials columns have no trial types; pass the trials CSV with
`-t/--trials` to recover them.
