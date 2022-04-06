# wavetracker

## py-scripts
- signal_tracker.py\
script to track signals of electric wavetype fish over time utilizing EOD frequency (fundmalentals) and signal strength 
across electrodes as tracking features.


- EOD_sorter.py
- signal_tracker_GUI.py

## parameters
- fundamentals: *2d-array*
- signatures: *3d-array*
- times: *1d-array*
    
## output

- fund_v.npy: *1d-array*\
das

- sign_v.npy: *2d-array*
- idx_v.npy: *1d-array*
- times.npy: *1d-array*
- ident_v.npy: *1d-array*
- spec.npy: *2d-array*
- meta.npy: *tuple*

- id_tag.npy

```py
import matplotlib.pyplot as plt
import numpy as np

fund_v = np.load('fund_v.npy', allow_pickle=True)
idx_v = np.load('idx_v.npy', allow_pickle=True)
ident_v = np.load('ident_v.npy', allow_pickle=True)
times = np.load('times.npy', allow_pickle=True)

fig, ax = plt.subplots()
ax.plot(times[idx_v[ident_v == 11]], fund_v[ident_v == 11], '.')
ax.set_xlabel('time [s]')
ax.set_ylabel('frequency [Hz]')
plt.show()
```

### (optional) High-resolution spectrograms
- fill_freqs.npy: *1d-array*\
  frequencies for high-res. spectrogram
- fill_times.npy:
  1d-array containing time stamps for high-res. spectrogram
- fill_spec.npy: (freq/time)
  2d-aray corresponding spectrogram
- fill_spec_shape.npy
  tuple: shape of fine spec
