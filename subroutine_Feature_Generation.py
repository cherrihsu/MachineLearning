import numpy as np
from subroutine_Load_QChem_Input_File import read_input
from subroutine_Load_QChem_Input_File import atom_list
from scipy.spatial import distance


def generate_CM(filename):
    ########### Read Q-Chem input file: XYZ positions and atom type of an pair ###########
    mol_array = read_input(filename)
    #print(mol_array)

    ########### Assign the input information as a class ###########
    mol_class = atom_list(mol_array)
    #print(mol_class.Natoms)
    #print(mol_class.atomtype)
    #print(mol_class.xyz)
    #print(mol_class.charge)
    #print(mol_class.mass)

    ########### Calculate distance between center of mass of molecular pair ###########
    Mass_Center=np.zeros((2, 3), float)
    no_atom_per_mol = int(mol_class.Natoms/2)
    mol_mass = np.sum(mol_class.mass[:no_atom_per_mol])
    ##### First molecule ####
    Mass_Center[0,0] = (np.sum(np.multiply(mol_class.xyz[:no_atom_per_mol,0], mol_class.mass[:no_atom_per_mol])))/mol_mass
    Mass_Center[0,1] = (np.sum(np.multiply(mol_class.xyz[:no_atom_per_mol,1], mol_class.mass[:no_atom_per_mol])))/mol_mass
    Mass_Center[0,2] = (np.sum(np.multiply(mol_class.xyz[:no_atom_per_mol,2], mol_class.mass[:no_atom_per_mol])))/mol_mass

    ##### Second molecule ####
    Mass_Center[1,0] = (np.sum(np.multiply(mol_class.xyz[no_atom_per_mol:,0], mol_class.mass[no_atom_per_mol:])))/mol_mass
    Mass_Center[1,1] = (np.sum(np.multiply(mol_class.xyz[no_atom_per_mol:,1], mol_class.mass[no_atom_per_mol:])))/mol_mass
    Mass_Center[1,2] = (np.sum(np.multiply(mol_class.xyz[no_atom_per_mol:,2], mol_class.mass[no_atom_per_mol:])))/mol_mass

    ##### Distance between molecular pair ####
    pair_dis = distance.euclidean(Mass_Center[0,:], Mass_Center[1,:])
    #print(pair_dis)


    ########### Calculate the diagonal element of Coulomb matrix ###########
    CM = np.zeros((mol_class.Natoms, mol_class.Natoms), float)
    cm_diagonal = 0.5*mol_class.charge**2.4
    np.fill_diagonal(CM, cm_diagonal)

    ########### Calculate the off-diagonal element of Coulomb matrix ###########
    for i in range(mol_class.Natoms):
      for j in range(i+1, mol_class.Natoms):
          # Calculate pairwise distance
          dst = distance.euclidean(mol_class.xyz[i], mol_class.xyz[j])
          CM[i,j] = mol_class.charge[i]*mol_class.charge[j]/dst
    #print(CM)


    ########### Generate different CM features ###########
    ##### CM_intra_inter_atom ####
    iu_idx = np.triu_indices(mol_class.Natoms)
    CM_intra_inter_atom = CM[iu_idx]
    #print(CM_intra_inter_atom)

    ##### CM_inter #####
    CM_inter = []
    for i in range(int(mol_class.Natoms/2)):
        CM_inter = np.concatenate([CM_inter, CM[i, int(mol_class.Natoms/2):mol_class.Natoms]])

    #return(CM_inter) 
    return(pair_dis, CM_intra_inter_atom)


def main():   
    import os
    ###### Set parameters #####
    QCmem_filename = 'Nap_35.inp'

    ###### Check the existence of the machine learning model and QChem input file #####
    cwd = os.getcwd()
    QChem_file_path = os.path.join(cwd, QCmem_filename)
    
    if not os.path.exists(QChem_file_path):
      raise ValueError('%s does not exist!' % QCmem_filename)
    
    print('Naphthalene pair from QChem file: %s' %(QCmem_filename))
    pair_dis, X = generate_CM(QChem_file_path)
    print(pair_dis)    
    
    
if __name__ == '__main__':
    main()    