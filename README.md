## eddycovariance_postprocessing
# Post Processing of eddy covariance data
This repository summarizes the Post-Processing methodology used for the eddy covariance data measured at Alerce Costero National Park and Biological Station Senda Darwin, Chile.
The aim of this methodology is to obtain filtered, screened and filled data from the flux towers.

# The initial level of this data is high frequency raw data that is obtained at a 20 Hz frequency

# The second step is the data processed in the software EddyPro in which the following corrections are applied:
**Eddypro - General Operations**

Fratini, G. and Matthias, M.: Towards a consistent eddy-covariance processing: an intercomparison of EddyPro and TK3, Atmospheric Measurement Techniques, 7(1), 2273–2281, 2014.

Foken, T. and Wichura, B.: Tools for quality assessment of surface-based flux measurements, Agricultural and forest meteorology, 78(1–2), 83–105, 1996.

Vickers, D. and Mahrt, L.: Quality control and flux sampling problems for tower and aircraft data, Journal of Atmospheric and Oceanic Technology, 14(3), 512–526, 1997.

**Eddypro - Tilt Corrections**

Wilczak, J. M., Oncley, S. P. and Stage, S. A.: Sonic anemometer tilt correction algorithms, Boundary-Layer Meteorology, 99(1), 127–150, 2001.

**Eddypro - WPL Terms**
 
Webb, E. K., Pearman, G. I. and Leuning, R.: Correction of flux measurements for density effects due to heat and water vapour transfer, Quarterly Journal of the Royal Meteorological Society, 106(447), 85–100, 1980.
 
Burba, G., Schmidt, A., Scott, R. L., Nakai, T., Kathilankal, J., Fratini, G., Hanson, C., Law, B., McDermitt, D. K., Eckles, R. and others: Calculating CO2 and H2O eddy covariance fluxes from an enclosed gas analyzer using an instantaneous mixing ratio, Global Change Biology, 18(1), 385–399, 2012.
 
Ibrom, A., Dellwik, E., Larsen, S. E. and Pilegaard, K.: On the use of the Webb–Pearman–Leuning theory for closed-path eddy correlation measurements, Tellus B, 59(5), 937–946, 2007.

**Eddypro - QC Flagging** 

Mauder, M. and Foken, T.: Impact of post-field data processing on eddy covariance flux estimates and energy balance closure, Meteorologische Zeitschrift, 15(6), 597–609, 2006.

**Eddypro - Footprint estimation (crosswind-integrated)** 

Kljun, N., Calanca, P., Rotach, M. and Schmid, H.: A simple parameterisation for flux footprint predictions, Boundary-Layer Meteorology, 112(3), 503–523, 2004.

**Eddypro - Spectral Corrections High Frequency** 

Ibrom, A., Dellwik, E., Flyvbjerg, H., Jensen, N. O. and Pilegaard, K.: Strong low-pass filtering effects on water vapour flux measurements with closed-path eddy correlation systems, Agricultural and Forest Meteorology, 147(3–4), 140–156, 2007.

**Eddypro - Spectral Corrections Low Frequency** 

Moncrieff, J., Clement, R., Finnigan, J. and Meyers, T.: Averaging, detrending, and filtering of eddy covariance time series, in Handbook of micrometeorology, pp. 7–31, Springer., 2004.

**Eddypro - Spectral Corrections Instruments Separation** 

Horst, T. and Lenschow, D.: Attenuation of scalar fluxes measured with spatially-displaced sensors, Boundary-layer meteorology, 130(2), 275–300, 2009.



# The last step in the data is this third level analysis which includes the following Workflow:

**Quality Screening, biomet merge and gapfill**

Isaac, Peter & Cleverly, Jamie & Mchugh, Ian & van Gorsel, Eva & Ewenz, Cacilia & Beringer, Jason. (2017). OzFlux data: Network integration from collection to curation. Biogeosciences. 14. 2903-2928. 10.5194/bg-14-2903-2017.

**Energy balance Residual Correction (not implemented yet)**

Matthias Mauder, Matthias Cuntz, Clemens Drüe, Alexander Graf, Corinna Rebmann, Hans Peter Schmid, Marius Schmidt, Rainer Steinbrecher. A strategy for quality and uncertainty assessment of long-term eddy-covariance measurements, Agricultural and Forest Meteorology, Volume 169, 2013, Pages 122-135, ISSN 0168-1923, https://doi.org/10.1016/j.agrformet.2012.09.006.

**u\* Threshold Detection**

Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N., Gilmanov, T., Granier, A., Grünwald, T., Havránková, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila, A., Loustau, D., Matteucci, G., Meyers, T., Miglietta, F., Ourcival, J.-M., Pumpanen, J., Rambal, S., Rotenberg, E., Sanz, M., Tenhunen, J., Seufert, G., Vaccari, F., Vesala, T., Yakir, D. and Valentini, R. (2005), On the separation of net ecosystem exchange into assimilation and ecosystem respiration: review and improved algorithm. Global Change Biology, 11: 1424-1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x

**Configurable MDS Gap Fill**

Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N., Gilmanov, T., Granier, A., Grünwald, T., Havránková, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila, A., Loustau, D., Matteucci, G., Meyers, T., Miglietta, F., Ourcival, J.-M., Pumpanen, J., Rambal, S., Rotenberg, E., Sanz, M., Tenhunen, J., Seufert, G., Vaccari, F., Vesala, T., Yakir, D. and Valentini, R. (2005), On the separation of net ecosystem exchange into assimilation and ecosystem respiration: review and improved algorithm. Global Change Biology, 11: 1424-1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x

**CO2 flux Partitioning**

Reichstein, M., Falge, E., Baldocchi, D., Papale, D., Aubinet, M., Berbigier, P., Bernhofer, C., Buchmann, N., Gilmanov, T., Granier, A., Grünwald, T., Havránková, K., Ilvesniemi, H., Janous, D., Knohl, A., Laurila, T., Lohila, A., Loustau, D., Matteucci, G., Meyers, T., Miglietta, F., Ourcival, J.-M., Pumpanen, J., Rambal, S., Rotenberg, E., Sanz, M., Tenhunen, J., Seufert, G., Vaccari, F., Vesala, T., Yakir, D. and Valentini, R. (2005), On the separation of net ecosystem exchange into assimilation and ecosystem respiration: review and improved algorithm. Global Change Biology, 11: 1424-1439. https://doi.org/10.1111/j.1365-2486.2005.001002.x
