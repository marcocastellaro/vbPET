# third_party

`vbPET_matlab/` (not in the repository) holds a copy of the original MATLAB
implementation of vbPET (UniPD lab code, G. Rizzo, EG and co-workers, 2015-2016),
used only as a numerical reference by `tests/test_matlab_compat.py`.

To run those tests, copy `UTILITY_VB/` of the MATLAB package into
`third_party/vbPET_matlab/UTILITY_VB/` or set `VBPET_MATLAB_DIR`. Without it
(and without cached fixtures) the tests are skipped.
