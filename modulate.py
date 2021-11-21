import numpy as np
import phonopy
from phonopy.interface.vasp import read_vasp,write_vasp
import phonopy.file_IO as file_IO
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import matplotlib.pyplot as plt
import time

start = time.time()
plt.rcParams.update({'font.size':22})
fig = plt.figure(figsize=(12,10))

## atomic positions
ncell = 20
bulk = read_vasp('POSCAR')
PrimitiveVectors = [[1,0,0],[0,1,0],[0,0,1]]
SuperCellVectors = [[4,0,0],[0,4,0],[0,0,4]]
phonon = Phonopy(bulk,SuperCellVectors,PrimitiveVectors)
force_constants = file_IO.parse_FORCE_CONSTANTS('FORCE_CONSTANTS')
phonon.set_force_constants(force_constants)

phonon.set_modulations(dimension=[[ncell,0,0],[0,ncell,0],[0,0,1]],phonon_modes=[[[0.5,0.5,0],0,500,0]]) # q-point, mode starting from 0, amp, phase 
cell0 = phonon.get_modulated_supercells()[0]
temp = np.copy(cell0.scaled_positions)
temp[:,2] %= 1
temp[:,2] /= ncell
cell0.scaled_positions = np.copy(temp)

## 20 20 20 perfect structure
total_cell = read_vasp('MPOSCAR-202020_perfect') ## I may edit this part later to make it created hear; you can just use Phonopy/VESTA to create perfect supercell
total_pos = cell0.scaled_positions
total_symbol = list.copy(cell0.symbols)

#amp = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1])*500 #M-R
amp = np.array([-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])*500 #50%
#amp = np.array([1,0.5,-1,1,0,1,1,0,-1,1,-1,1,-1,1,0.5,1,-1,1,-1])*500 # temp1
#amp = np.array([0,-0.5,1,0,0,-0.5,1,0,0.5,-1,-1,1,0.5,1,0,1,0,1,-0.5])*500 # temp2
#amp = np.array([0,1,-1,0.5,0.5,-0.5,1,1,-1,-1,0,1,-0.5,0,0,-1,0,-1,1])*500 # temp3
#amp = np.array([1,0.5,0,0,-1,-1,0,0,-1,1,0.5,1,1,0,-1,-1,0,0,-1])*500 # temp4
#amp = np.array([7.31375151e+00,  1.00000000e+00,  2.96261051e+00,  1.00000000e+00,  2.00000000e+00,  1.00000000e+00,  1.50952545e+00, 1.00000000e+00,  1.15838444e+00,  1.00000000e+00,  8.41615560e-01,  1.00000000e+00,  4.90474551e-01,  1.00000000e+00,  1.17083638e-15,  1.00000000e+00, -9.62610506e-01,  1.00000000e+00, -5.31375151e+00,  1.00000000e+00])*75 #

## Create a 20 layer supercell with each layer's octahedral rotation. Different layer could have different rotation amplitude or phase (sign of amplitude).
## Octahedral rotation is created by phonon mode at M point (0.5,0.5,0) for a simple cubic perovskite

for i in range(ncell-1):
    phonon.set_modulations(dimension=[[ncell,0,0],[0,ncell,0],[0,0,1]],phonon_modes=[[[0.5,0.5,0],0,amp[i],0]]) # q-point, mode starting from 0, amp, phase 
    cell = phonon.get_modulated_supercells()[0]

    temp_pos = np.copy(cell.scaled_positions)
    temp_pos[:,2] %= 1 # to scale the fractional coordinate in a single layer to be 20 layers
    temp_pos[:,2] /= ncell
    cell.scaled_positions = np.copy(temp_pos)

    total_pos = np.concatenate((total_pos,cell.scaled_positions + (i+1)*np.array([0,0,1/ncell])))
    total_symbol.extend(cell.symbols)

total_cell.scaled_positions = np.copy(total_pos)
total_cell.symbols = total_symbol
write_vasp('MPOSCAR_modulated',total_cell) # you can visualize structure by VESTA
print ('Finished Modulation')
positions = total_cell.scaled_positions #5000*3 total, 3000 Br, 1000 Cs and Pb

## Q mesh
Qstep = 1/ncell
Qxz = np.mgrid[-4:4+Qstep:Qstep, -4:4+Qstep:Qstep].reshape(2,-1).T
Qxz = Qxz.reshape((ncell*8+1,ncell*8+1,2)) # the cell is 20 20 20

## Neutron scattering length
b_Br = 6.795 
b_Cs = 5.42
b_Pb = 9.405

##
print ('Start FT (this may take a while...)')
## Fourier transform for H 0.5 L plane
four = np.zeros((ncell*8+1,ncell*8+1))
for i in range(ncell*8+1):
    for j in range(ncell*8+1):
        temp = 0
        Q = np.array([Qxz[i,j][0],0.5,Qxz[i,j][1]])*ncell # 10 10 10 cells
        # Br
        temp = b_Br*np.sum(np.exp(2*np.pi*1j*np.inner(Q,positions[0:24000])))
        # Cs 
        temp += b_Cs*np.sum(np.exp(2*np.pi*1j*np.inner(Q,positions[24000:32000])))
        # Pb 
        temp += b_Pb*np.sum(np.exp(2*np.pi*1j*np.inner(Q,positions[32000:40000])))
        four[i,j] = np.abs(temp)**2

print ('Finish FT')
four = np.rot90(four)
np.savetxt('CPB_four.txt',four) ## you could plot this later by yourself
plt.imshow(np.log10(four),vmin=5.5,vmax=7.5,cmap='viridis')
xpos = np.linspace(0,160,9)
plt.xticks(xpos,np.linspace(-4,4,9))
ypos = np.linspace(160,0,9)
plt.yticks(ypos,np.linspace(-4,4,9))
plt.colorbar()
plt.xlabel('H 0 0')
plt.ylabel('0 0 L')
plt.title('CPB H 0.5 L')

plt.savefig('CPB_MR_layer_diffuse.png')
end = time.time()
print (end-start)
#plt.show()
