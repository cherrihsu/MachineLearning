# The unit of electronic coupling predicted by the machine learning model is electron volt (eV)

import numpy as np
import os
import torch
from torch.autograd import Variable
from subroutine_NN_Architecture import Model
from subroutine_Feature_Generation import generate_CM

###### Set parameters #####
QCmem_filename = 'Nap_35.inp' #Specify QChem input file
Model_name = 'ANN-3.pth'  #Specify the artificial neural network model
device = "cpu" # cuda or cpu


def main():     
    ###### Check the existence of the machine learning model and QChem input file #####
    cwd = os.getcwd()
    ML_model_path = os.path.join(cwd, Model_name)
    QChem_file_path = os.path.join(cwd, QCmem_filename)
    
    if not os.path.exists(ML_model_path):
      raise ValueError('%s does not exist!' % Model_name)
    if not os.path.exists(QChem_file_path):
      raise ValueError('%s does not exist!' % QCmem_filename)
    
    print('Machine learning model: %s' %(Model_name))
    print('Naphthalene pair from QChem file: %s' %(QCmem_filename))


    ###### Load the xyz coordinates of naphthalene pair from QChem input file & generate the corresponding Coulomb matrix #####
    pair_distance, X = generate_CM(QChem_file_path)
    print('Distance between naphthalene pair: %s [angstrom]' %(pair_distance))
    if (pair_distance>7.0):
        print("Warning: Distance between naphthalene pair is larger than 7 [angstrom].")
        print("         The predicted coupling value may not be accurate enough.")


    ###### Check the dimenssion of input feature of Coulomb matrix #####
    D_feature = X.shape[0]



    ###### Change into PyTorch datat format #####
    X = X.reshape(1, D_feature)
    X = Variable(torch.from_numpy(np.array(X)))

    if device == "cuda":
      print('Data are processed on GPU.')
      X = X.cuda().float()
    elif device == "cpu":
      print('Data are processed on CPU.')
      X = X.float()
    else:
      raise ValueError('Invalid device name %s, please type cuda or cpu!' % device)


    ###### Loading machine learning model #####
    model = Model(D_feature)  
    model.load_state_dict(torch.load(ML_model_path, map_location=torch.device(device)))
    model.to(torch.device(device))
    

    ###### Predict electronic coupling by machien learning Model #####
    model.eval()
    Y_predict = model.forward(X).detach()
    print('Predicted electronic coupling: %.2f [meV]' %(Y_predict.cpu().numpy()[0,0]*1000))
    
    if (QCmem_filename == "ANN-3.pth"):
        scale_factor = 1.17
        print('Predicted electronic coupling after rescaling: %.2f [meV]' %(Y_predict.cpu().numpy()[0,0]*1000*scale_factor))
    elif (QCmem_filename == "ANN-2.pth"):
        scale_factor = 1.19
        print('Predicted electronic coupling after rescaling: %.2f [meV]' %(Y_predict.cpu().numpy()[0,0]*1000*scale_factor))

if __name__ == '__main__':
    main()



