# MachineLearning
The artificial neural networks predict charge transport coupling of a naphthalene dimer. All the information of the models is published in The Journal of Chemical Physics by Chun-I Wang, Ignasius Joanito, Chang-Feng Lan, and Chao-Ping Hsu. We provided five well-trained models that are built up by PyTorch and are denoted as ANN-1, ANN-2, ANN-3, ANN-4, and ANN5 in the article and the following introduction. Four example files are also provided for the test of our ML models. They are four pairs of naphthalene molecules that follow the format of QChem input file. For ANN-2 and ANN-3, we also provide the coupling value that is rescaled by an scaling factor, and that can reach the accuracy of high-level ab initio calculation.

Please download all the python source codes, the PyTorch model files ,and the testing files:
1. ML_Prediction.py (main source code)
2. subroutine_Feature_Generation.py (subroutine)
3. subroutine_Load_QChem_Input_File.py (subroutine)
4. subroutine_NN_Architecture.py (subroutine)
5. ANN-1.pth,  ANN-2, ANN-3.pth, ANN-4.pth and  ANN-3.pth (PyTorch model files)
6. Nap_35.inp, Nap_40.inp, Nap_45.inp, and Nap_50.inp (testing files)

########## Software/Libraries Requirement ##########
1. Python 3.6
2. Scikit Learn version 0.19.2
3. Numpy version 1.16.2
4. PyTorch version 1.3.1
5. Scipy version 1.2.1

########## Performing ML with testing Files ##########
1. Specify the testing file in ML_Prediction.py
   e.g., QCmem_filename = 'Nap_35.inp'
2. Specify the ML model (ANN-2 or ANN-3) in ML_Prediction.py
   e.g., Model_name = 'ANN-3.pth' 
3. Save and run "ML_Prediction.py" 

########## Performing ML with new data ##########
1. Make sure the order of atoms of a new pair in the new QChem in put file is in the same order of atom sequence as the provided testing files!
2. Specify the testing file in ML_Prediction.py
   e.g., QCmem_filename = 'Nap_35.inp'
3. Specify the ML model (ANN-2 or ANN-3) in ML_Prediction.py
   e.g., Model_name = 'ANN-3.pth' 
4. Save and run "ML_Prediction.py" 
   The unit of electronic coupling predicted by the machine learning model is electron volt (eV)
