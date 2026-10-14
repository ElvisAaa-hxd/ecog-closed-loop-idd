# ecog-closed-loop-idd

Closed-loop seizure detection and responsive stimulation algorithm for an implantable ECoG device.

The algorithm runs on a Seeed XIAO RP2040 with an ADS1220 ADC (SPI, DRDY interrupt), filters the signal into the beta band, tracks an adaptive detection threshold from the resting baseline, locks the threshold during a detected seizure, and emits a therapeutic output whose duration scales with the detected burst energy.

## Repository layout

```
firmware/Brain_ECoG_Serial.ino   on-device implementation (ADS1220 + RP2040)
analysis/stream_fft_judge.py     offline replica + real-time streaming visualizer (pygame)
analysis/offline_calibration.py  headless calibration harness (filter/threshold/lock/duration grid)
data/EPILEPSY.csv                example ECoG recording (600 samples/s, two columns: time, amplitude in mV)
```

## Algorithm

1. Beta-band biquad band-pass filter, centre 16.75 Hz, Q = 2 (passband 12.6-20.9 Hz).
2. Seizure score S = variance over a short window of 20 samples.
3. Adaptive threshold = D x variance over a long window of 1000 samples (resting baseline tracker). The threshold is locked at its crossing value for 30 s after the first supra-threshold sample, then re-adapts.
4. Detection when S exceeds the threshold at least T times within a 600-sample (1 s) decision epoch.
5. Therapeutic duration = R x sum of squared deviations over the short window, clamped between 0.1 s and 10 s.

Final parameter values (identical in firmware and the Python replica):

| Parameter | Value | Role |
|---|---|---|
| D | 22.0 | detection sensitivity: threshold stays about fourfold above the largest resting excursion |
| T | 5 | trigger persistence: rejects isolated transients of fewer than five supra-threshold samples |
| R | 0.0012 | dosage control: duration proportional to detected burst energy (mean onset score 134 mV^2 maps to about 3 s) |
| short window | 20 samples | detection window (about 33 ms at 600 samples/s) |
| long window | 1000 samples | baseline window |
| lock | 30 s | threshold freeze during an episode |

Sampling: the reference recording is 600 samples/s; the device firmware acquires at 1000 samples/s with the same sample-count windows. Signal amplitudes are ADC-referred mV.

## Quick start

```bash
pip install numpy pygame
python analysis/stream_fft_judge.py        # streaming visualizer on data/EPILEPSY.csv
python analysis/offline_calibration.py --final --D 22 --T 5 --lock 30 --R 0.0012
```

The firmware targets the Arduino core for XIAO RP2040 and uses the Protocentral_ADS1220 library.

## Release

v1.0.0 - calibrated parameter set used for the reported experiments.
