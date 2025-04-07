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

```bash
mesoscopy register mark-landmarks /path/to/example-recording.nwb
```



```bash
mesoscopy register landmarks --template-points example-recording_landmarks.csv /path/to/example-recording.nwb
```


## Extract area responses

```bash
mesoscopy process area-responses /path/to/example-recording.nwb
```


## Next steps
