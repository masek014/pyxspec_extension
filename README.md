# pyxspec_extension
A Python implementation to interface with PyXSPEC.
It is a wrapper making it (hopefully) easier to interace with XSPEC via PyXSPEC.
Currently, one can load data, select models, fit the model to the data, and plot the fitted model.

Notable additions to the standard PyXSPEC package are:
- A series of archive classes designed to make parameter storage, retrieval, and organization easier.
- The ability to *add* a model component to an existing model; this is not possible with vanilla PyXSPEC.
- A plotting class that allows one to plot the resulting fit; however, this class is currently designed specifically for my use with *NuSTAR*, but it is something that I will generalize.

See `examples` on how to perform an isothermal fit with increasing complexity: with a pileup model, then with a background model (with its own pileup model).
The example flare is from 2018-05-29 16:06 UTC.