# Typical workflow

## Convert recording file to NWB format

```bash
mesoscopy convert /path/to/example-recording.h5
```

!!! note
    See also our guide to [converting video files to HDF5](how-to/convert-video-to-h5.md) if you're recording videos in AVI or MP4 format.

## Inspect raw data

```bash
mesoscopy inspect /path/to/example-recording.nwb
```

## Preprocess to correct for haemodynamics and extract ∆F signal

```bash
mesoscopy preprocess /path/to/example-recording.nwb
```

## Register to the Allen Brain Atlas

First mark the anatomical landmarks on the recording. This opens the napari landmark GUI, seeded
with a point per landmark: drag each one onto its anatomical location, then press **Save and Close**.

```bash
mesoscopy register label /path/to/example-recording.nwb
```

This writes `example-recording_landmarks.csv` to the output directory (`-o`, the current directory
by default), holding each landmark's `(x, y)` position in the pixel space of the ∆F/F series.

Then warp the recording onto the atlas. The landmarks file is found automatically if it sits next to
the recording or in the output directory; pass `-r/--recording-points` to point at it explicitly.

```bash
mesoscopy register landmarks /path/to/example-recording.nwb
```

The registered frames are written in Allen CCF template space at the atlas's own dimensions, which
is what `mesoscopy process area-responses` expects. Use `--output-width` / `--output-height` only if
you are registering onto a different template.

!!! note
    `-t/--template-points` supplies the *template* landmarks being registered onto, not your
    recording's landmarks. Leave it unset to use the Allen CCF landmarks that ship with mesoscopy.

Registration reports how far each landmark ends up from its template position, and warns if the fit
is poor:

```
Estimating transform from 9 landmarks...
Landmark fit: RMSE 2.28 px, worst is 'rFP' at 3.94 px (in template pixels).
```

To check the alignment visually, generate the QA report for the registered HDF5 file written by the
step above — its path is echoed as `Saved registered frames at ...`:

```bash
mesoscopy report /path/to/example-recording_registered.h5
```

## Extract area responses

```bash
mesoscopy process area-responses /path/to/example-recording.nwb
```

## Next steps
