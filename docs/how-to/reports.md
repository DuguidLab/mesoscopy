# Reports

`mesoscopy report` writes a self-contained HTML report for a pipeline output and picks the report from the
filename suffix. The report lands next to the input, or under `-o`.

```bash
mesoscopy report /path/to/recording_preprocessed.h5
mesoscopy report /path/to/recording_registered.h5
mesoscopy report /path/to/recording_smoothed_regions_event-cueonset_perievent.csv -o /path/to/reports
```

Reports need an internet connection to render and are otherwise self-contained.

## Preprocessing report

For a `*_preprocessed.h5` written by `preprocess`. It shows the session metadata, the automated QA checks
(histogram separation, timestamp consistency and jumps, noise, SNR, bleaching), the timestamp series and its
jumps, the per-frame statistics used for channel separation and the resulting frame assignment, the mean ∆F/F of
each channel, and the haemodynamics-corrected ∆F/F with its projections and an example frame.

## Registration report

For a `*_registered.h5` written by `register landmarks`. It shows the recording landmarks against the template
landmarks before and after the affine transform, and a registered frame with the template landmarks overlaid.

## Peri-event report

For a `*_perievent.csv` written by `process peri-event`. It shows each region's response around the event as the
mean ± 95% CI across trials or as individual trials, filtered by trial type, one region at a time and as a grid of
every region. Choose the region from the dropdown, by clicking the Allen CCF top view, or by clicking a panel in
the grid.

When the `*_metrics.csv` and `*_metrics-session.csv` from `process metrics` sit next to the input, the report adds
a metrics section. It marks the onset, peak and offset on the traces with the across-trial spread of each, plots
per-trial box plots for the selected region, colours the atlas by a session metric, and lists the session table.
The atlas and the session table follow the [trial group](metrics.md#trial-groups) chosen above the table, and for
go/no-go sessions the table also shows the reliability metrics. Peri-event files written without the trials
columns have no trial types. Pass the trials CSV with `-t/--trials` to recover them.

When the `*_connectivity.csv` from `process connectivity` sits next to the input, the report adds a
[connectivity](connectivity.md) section. It shows the chosen metric for the chosen trial group as a region by
region heatmap, and the trial group follows the selector above the session table. The transfer entropy and its
z-score are directed, from the row region to the column region. Click a cell to select its row region in the rest
of the report.
