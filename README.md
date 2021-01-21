# Developed Software simulating collagen matrix deformations. 

Code developed and used for the simulations of the paper: Cells exploit a phase transition to mechanically remodel the fibrous extracellular matrix, [available on arxiv](https://arxiv.org/abs/1905.11246).



### Software requirements

* FEniCS up to version 2017.2.0, ([installation instructions](https://fenicsproject.org/download/)). It is recommended to use FEniCS on Docker.


### Executing program

* Adding the higher order term, i.e. the regularization parameter, requires at least second order Lagrange elements. 
Se file  example_epsilon.py, can be executed typing 
```
mpirun -np N python example_epsilon.py
```
where N is the number of processes. **The higher gradient parameter ε should be smaller than the mesh size.**

### Resulting files
Results are stored in the folder 'results/vtkFiles/'. This folder contains:
* dens.pvd (vtk format): The density (1/detF), in the deformed configuration.

* detF (vtk format):  The determinant, the volume ration deformed/reference, 
of the deformation matrix in the reference configuration.

* u.pvd (vtk format): The computed displacement in the reference configuration.

* saved_functions/reference_funs.h5  (hdf5 format): Contains the reference 
dolfin.cpp.mesh object (the mesh) and the computed solution in the dolfin.functions.function object
(the computed displacement that minimizes the energy).
Read this data using the file 'read_hdf5_data.py'.


## Author
 
Georgios Grekas  

## Help

For any advise for common problems or issues please contact the author.

