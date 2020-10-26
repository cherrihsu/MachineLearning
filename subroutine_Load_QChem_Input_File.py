import numpy as np

def read_input(filename):
    # Open inputfile
    f = open(filename, "r")
    full_content = f.readlines()
    f.close()
    
    # Get the basic structure 
    block_index = [[], []]
    blocks = []
    switch = 1
    
    for num, line in enumerate(full_content):
        if "$" in line:
            if switch == 1:
                (block_index[0]).append(num)
                blocks.append((line.strip().lower())[1:])
                switch = 0
            elif switch == 0:
                (block_index[1]).append(num)
                switch = 1
    
    # Create inputfile object
    mol_array = []
    for i, k in enumerate(blocks):
        k = k.split('!')[0].strip()
        if (k == "molecule"):
            content = full_content[block_index[0][i] + 1:block_index[1][i]]
            for line in content[:]:
                atom_input = line.strip().split()
                if len(atom_input) < 4:
                    continue
                atom = atom_input[0].capitalize()
                x = atom_input[1]
                y = atom_input[2]
                z = atom_input[3]
                mol_array.append([atom, x, y, z])
    mol_array = np.array(mol_array)
    
	
    return mol_array



class atom_list():
    def __init__(self, mol_array=[]):
        self.mol_array = mol_array
        self.Natoms = mol_array.shape[0]
        self.atomtype = []
        self.xyz = []
        self.charge = []
        self.mass = []
        for i in range(self.Natoms):
            x = mol_array[i][1]
            y = mol_array[i][2] 
            z = mol_array[i][3]
            self.xyz.append(np.array([float(x), float(y), float(z)]))

            self.atomtype.append(mol_array[i][0])
            if (mol_array[i][0] == "C" ):
                self.charge.append(float(6.0000000))
                self.mass.append(float(12.011))
            elif (mol_array[i][0] == "H" ):
                self.charge.append(float(1.0000000))
                self.mass.append(float(1.00784))
            elif (mol_array[i][0] == "N" ):
                self.charge.append(float(7.0000000))
                self.mass.append(float(14.007))
            else:
                raise ValueError('The charge of atom %s has not been defined in this class.' % mol_array[i][0])
        self.charge = np.array(self.charge)
        self.xyz = np.array(self.xyz)        
        