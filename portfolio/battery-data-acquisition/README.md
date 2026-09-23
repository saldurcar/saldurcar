# Battery Experimental Data Acquisition

Public portfolio version of a Python workflow developed to support controlled lithium-ion battery thermal experiments.

## What the research workflow handles

- Serial acquisition from an embedded controller
- Multiple temperature channels
- Voltage and current measurements
- 1 Hz synchronized experiment records
- Thermal-rate calculation (dT/dt)
- State-of-charge estimates
- Image/frame naming and synchronization
- CSV logging
- Live engineering visualization
- Threshold-based temperature alarms

## Public-code policy

The complete laboratory configuration, raw experimental dataset, local paths, hardware identifiers, and unpublished experiment-specific parameters are not distributed here.

The included code is a **sanitized representative implementation** showing the acquisition architecture without exposing private research data.

## Skills demonstrated

Python · serial data acquisition · multithreading · sensor fusion · time-series logging · NumPy · engineering visualization · experimental instrumentation
