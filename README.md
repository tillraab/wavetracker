# Wavetracker

Scripts, APIs, and algorithms used to analyze electrode-array recordings of wave-type
electric fish. This includes (i) the extraction of individual electric signals from the
corresponding raw data, (ii) the tracking of individual fish signals for whole recordings, 
and (iii) the post-processing to eliminate and correct tracking errors.

**Purpose of this fork:** Add gui interface to save the currently displayed plot including
metadata.

## trackingGUI.py
GUI interface combining (i) the extraction of electric wave-type signals in  electrode-array 
recording of electric fish using the 
[ThunderFish-package](https://github.com/bendalab/thunderfish) and (ii) 
 tracking of individual specific signals using the **signal_tracker algorithm** (see below).
Settings for **Spectrogram** analysis and signal identification using peak detection (**harminic group** detection) can be customized
according to the requirements of a dataset. 

**Run-option:**\
Multiple files can be analyzed consecutively by adding them to a queue using  the **open** 
button before **run**ing the application. The **Auto Save** option is generally suggested 
and should only be left unchecked for debugging or test purposes. This analysis option delivers all **required** files for post-processing using the **EODsorter-GUI**.

**Calc. fine spec-option:**\
The analysis option generates a high resoultion spectrogram of the analyzed data and generates 
the **optional** files for post-processing using the **EODsorter-GUI**.

NOTE: the resulting files are rather large and are required to be loaded correctly 
(see **examples**) in order to not overwrite the file containting the high resolution spectrogram.


## signal_tracker.py
Script to track signals of electric wavetype fish over time utilizing EOD frequency (fundmalentals) and signal strength 
across electrodes (signatures) as tracking features. The input of these funcations are based on the output of **trackingGUI.py** 
which utilizes spectrogram analysis and peak detection algorithms comprized in the [ThunderFish-package](https://github.com/bendalab/thunderfish)
and used to detect individual signals of wave-type electric fish and electrode-array recordings.


**freq_tracking_v5(fundamental, signatures, times, visualize=True)**\
Main tracking algorithm utilizing frequency and signal power across recording electrodes to
identify signals of multiple different wave-type electric fish and track them over time 
according to highest-signal similarity my means of frequency and spatial field property differences.

**Parameters:**

- fundamentals: *2d-array*\
Contains for each timestep the detected fundamental frequencies that shall be tracked.
  

- signatures: *3d-array*\
Contains signal powers across recording electrodes for each signal detected in *fundamenals*.


- times: *1d-array*\
Timestamps of the underlying spectrogram which provided *fundamentals* and *spectrograms*
  

- visualizd: *bool*\
Optional visual representation of the current tracking process. Slows down tracking process tremendously.

**Output:**

- fund_v: *1d-array*\
Contains frequencies of signals regarded in the tracking process.
  

- sign_v.npy: *2d-array*\
Signal powers across recording electrodes for signals regarded in the tracking process.
  

- idx_v.npy: *1d-array*\
Time indices for signals regarded in the tracking process.
  

- times.npy: *1d-array*\
Times in seconds referred to in *idx_v*   


- ident_v.npy: *1d-array*\
Contains the assigned identities (float) for all signals regarded in the tracking process. NaN-vallues refer to signals 
  that have not be assigned to an identity.


## EODsorter.py

GUI interface user to post-process tracked signals obtained from **signal_tracker.py**.
In order to enable processing of a dataset, associated files needs to be stored in one folder 
(which is selected in the "open"-Dialog). 

**Required files:**
- fund_v: *1d-array*\
Contains frequencies of signals regarded in the tracking process.
  

- sign_v.npy: *2d-array*\
Signal powers across recording electrodes for signals regarded in the tracking process.
  

- idx_v.npy: *1d-array*\
Time indices for signals regarded in the tracking process.
  

- times.npy: *1d-array*\
Times in seconds referred to in *idx_v*   


- ident_v.npy: *1d-array*\
Contains the assigned identities (float) for all signals regarded in the tracking process . NaN-vallues refer to signals 
  that have not be assigned to an identity.


- meta.npy: *tuple*\
First and last second of the analized dataset.


- spec.npy: *2d-array*\
Low resolution spectrogram used for visual feedback when handling and processing tracked data.
  The resolution of the spectrogram is selected to fit the size of a 15" monitor. In the frequency domain it extends 
  from 0 to 1000Hz, in the time domain from the first to the last analyzed second infered to in **meta.npy**.

**Optional files:**
- fill_spec.npy: *2d-array*\
High-resolution spectrogram that can partially be loaded to gain detailed spectral insights into the evaluated data.
  

- fill_freqs.npy: *1d-array*\
Frequencies for high-resesolution spectrogram.
  

- fill_times.npy: *1d-array*\
Time stamps for high-resolution spectrogram.
  

- fill_spec_shape.npy: *tuple*\
Shape of fine spec.

## Examples:

**Display tracked EOD frequency traces:**
```py
import matplotlib.pyplot as plt
import numpy as np

fund_v = np.load('fund_v.npy', allow_pickle=True)
idx_v = np.load('idx_v.npy', allow_pickle=True)
ident_v = np.load('ident_v.npy', allow_pickle=True)
times = np.load('times.npy', allow_pickle=True)

ids  = np.unique(ident_v[~np.isnan(ident_v)])

fig, ax = plt.subplots()
ax.plot(times[idx_v[ident_v == ids[0]]], fund_v[ident_v == ids[0]], marker='.')
ax.set_xlabel('time [s]')
ax.set_ylabel('frequency [Hz]')
plt.show()
```

**Load and display fine spectrogram**
```py
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.powerspectrum import decibel

fill_freqs = np.load('fill_freqs.npy', allow_pickle=True)
fill_times = np.load('fill_times.npy', allow_pickle=True)
fill_spec_shape = np.load('fill_spec_shape.npy', allow_pickle=True)
fill_spec = np.memmap('fill_spec.npy', dtype='float', mode='r', 
                      shape=(fill_spec_shape[0], fill_spec_shape[1]), order = 'F')

f0, f1 = 400, 420 # frequency limitation
t0, t1 = 100, 200 # time limitation

f_mask = np.arange(len(fill_freqs))[(fill_freqs >= f0) & (fill_freqs <= f1)]
t_mask = np.arange(len(fill_times))[(fill_times >= t0) & (fill_times <= t1)]

fig, ax = plt.subplots()
ax.imshow(decibel(fill_spec[f_mask[0]:f_mask[-1], t_mask[0]:t_mask[-1]][::-1]), 
          extent=[t0, t1, f0, f1], aspect='auto', vmin = -100, vmax = -50, alpha=0.7, 
          cmap='jet', interpolation='gaussian')
plt.show()
```
