# Global Options

ADCME manages certain algorithm hyperparameters using a global option `ADCME.option`. 

```@docs
Options
OptionsSparse
OptionsNewtonRaphson
OptionsNewtonRaphson_LineSearch
reset_default_options
```

The current options and their default values are

```julia
using PrettyPrint
using ADCME
pprint(ADCME.options)
```

```text
ADCME.Options(
  sparse=ADCME.OptionsSparse(
    auto_reorder=true,
    solver="SparseLU",
  ),
  newton_raphson=ADCME.OptionsNewtonRaphson(
    max_iter=100,
    verbose=false,
    rtol=1.0e-12,
    tol=1.0e-12,
    LM=0.0,
    linesearch=false,
    linesearch_options=ADCME.OptionsNewtonRaphson_LineSearch(
      c1=0.0001,
      ρ_hi=0.5,
      ρ_lo=0.1,
      iterations=1000,
      maxstep=9999999,
      αinitial=1.0,
    ),
  ),
)
```

