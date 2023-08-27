# SVM classifier in AMPL
This project implements different versions of a SVM classifier in AMPL
## Authors
- Benet Ramió
- Pau Amargant
  
## Usage
In order to run the project, Python 3.11 and the following packages have been used. Note that it has been run using Linux:
- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- plotly   
- seaborn
In order to fit the models AMPL throught amplpy is used. Amplpy can be installed using the following command:
```
pip install amplpy
python -m amplpy.modules install ipopt
python -m amplpy.modules activate <LICENSE_KEY>
```
An AMPL license is required to run the project. A temporal LICENSE_KEY for limited usage is 
`570845f4-0a14-4e32-824f-7b66b06a66b0`.

## Replicating the results
In order to replicate the results the Jupyter Notebooks can be run. Note that the Linearly separable notebook includes a step with a long running time and can be skipped loading the results from a pickle file.
