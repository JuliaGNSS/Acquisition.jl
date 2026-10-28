# API Reference

## Acquisition Functions

```@docs
acquire
acquire!
acquire_stream
acquire_stream!
plan_acquire
cfar_threshold
is_detected
get_num_cells
generate_test_signal
recommend_sampling_freqs
is_valid_sample_rate
sample_rate_range
```

## Types

```@docs
AcquisitionResults
AcquisitionPlan
AD9361ClockPlan
Acquisition.PlanInUseError
Acquisition.SamplingFreqRecommendation
Acquisition.AbstractSDRClockPlan
```

## Noise Estimators

```@docs
OppositeRowNoiseEstimator
GlobalMeanNoiseEstimator
Acquisition.AbstractNoiseEstimator
```

## Index

```@index
```
