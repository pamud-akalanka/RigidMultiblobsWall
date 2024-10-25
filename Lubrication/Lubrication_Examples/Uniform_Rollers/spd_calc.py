import numpy as np


""" # Define the filename
filename = 'problem_positions.clones'

# Read the data from the file
try:
    positions = np.loadtxt(filename)

    # Check if the data has the correct shape
    if positions.ndim == 2 and positions.shape[1] == 7:
        print(f"Data successfully read from {filename}.")
        print("First 5 positions:")
        print(positions[:5])
    else:
        print(f"Unexpected data shape: {positions.shape}. Expected an Nx3 array.")

except IOError:
    print(f"Error: The file {filename} does not exist or cannot be read.")
except ValueError as e:
    print(f"Error: There was a problem reading the data from the file. {e}")
 """

import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
try:
    import pickle as cpickle
except:
    try:
        import cpickle
    except:
        import _pickle as cpickle
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from sksparse.cholmod import cholesky
#import matplotlib.pyplot as plt

# Find project functions
found_functions = False
path_to_append = ''
sys.path.append('../../')
while found_functions is False:
    try:
        from Lub_Solver import Lub_Solver as LS
        from stochastic_forcing import stochastic_forcing as stochastic
        from mobility import mobility as mb
        from body import body
        from read_input import read_input
        from read_input import read_vertex_file
        from read_input import read_clones_file
        from read_input import read_slip_file
        import general_application_utils
        import multi_bodies_functions

        found_functions = True
    except ImportError:
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProjected functions not found. Edit path in multi_bodies.py')
            sys.exit()

if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description='Run a multi-body simulation and save trajectory.')
    parser.add_argument('--input-file', dest='input_file', type=str,
                        default='data.main', help='name of the input file')
    parser.add_argument('--print-residual', action='store_true',
                        help='print gmres and lanczos residuals')
    args = parser.parse_args()
    input_file = args.input_file

    # Read input file
    read = read_input.ReadInput(input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    output_name = read.output_name
    structures = read.structures
    print(structures)
    structures_ID = read.structures_ID

    # Copy input file to output
    subprocess.call(["cp", input_file, output_name + '.inputfile'])

    # Set random generator state
    if read.random_state is not None:
        with open(read.random_state, 'rb') as f:
            np.random.set_state(cpickle.load(f))
    elif read.seed is not None:
        np.random.seed(int(read.seed))

    # Save random generator state
    with open(output_name + '.random_state', 'wb') as f:
        cpickle.dump(np.random.get_state(), f)

    # Create rigid bodies
    bodies = []
    body_types = []
    body_names = []
    for ID, structure in enumerate(structures):
        print('Creating structures = ', structure[1])
        # Read vertex and clones files
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(
            structure[1])
        # Read slip file if it exists
        slip = None
        if(len(structure) > 2):
            slip = read_slip_file.read_slip_file(structure[2])
        body_types.append(num_bodies_struct)
        body_names.append(structures_ID[ID])
        # Create each body of type structure
        for i in range(num_bodies_struct):
            b = body.Body(
                struct_locations[i], struct_orientations[i], struct_ref_config, a)
            b.ID = structures_ID[ID]
            # Calculate body length for the RFD
            if i == 0:
                b.calc_body_length()
            else:
                b.body_length = bodies[-1].body_length
            # Append bodies to total bodies list
            bodies.append(b)
    bodies = np.array(bodies)

    # Set some more variables
    num_of_body_types = len(body_types)
    num_bodies = bodies.size
    num_particles = len(bodies)
    Nblobs = sum([x.Nblobs for x in bodies])

    cutoff = read.Lub_Cut

    L = read.periodic_length
    phi = 0.4
    Lphi = np.sqrt(np.pi*(a**2)*num_particles/phi)
    #L = np.array([Lphi, Lphi, 0])

    n_steps = read.n_steps
    n_save = read.n_save
    dt = read.dt

    print(L)

    for b in bodies:
        for i in range(3):
            if L[i] > 0:
                while b.location[i] < 0:
                    b.location[i] += L[i]
                while b.location[i] > L[i]:
                    b.location[i] -= L[i]

    firm_delta = read.firm_delta
    debye_length_delta = 2.0*a*firm_delta/np.log(1.0e1)
    repulsion_strength_delta = read.repulsion_strength_firm

    LSolv = LS(bodies, a, eta, cutoff, L, debye_length=firm_delta)
    LSolv.dt = dt
    LSolv.kT = read.kT
    LSolv.tolerance = read.solver_tolerance

    multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(
        read.blob_blob_force_implementation)
    multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(
        read.body_body_force_torque_implementation)

    import time
    t0 = time.time()
    LSolv.Set_R_Mats()
    dt1 = time.time() - t0
    print(("Make R mats time : %s" % dt1))



    r_vecs = np.array([1.0* b.location for b in bodies])

    print ("Rvecs", r_vecs)

    delta_R = LSolv.Delta_R

    print(("Delta_R :", delta_R))


    from scipy.sparse import rand
    from scipy.sparse.linalg import eigsh

    # Define matrix dimensions
    #N = 2370

    # Generate a random sparse matrix (using a specific density for demonstration)
    # Adjust the density as needed; here it's set to 0.01 (1% non-zero entries)
    #density = 0.01
    sparse_matrix = delta_R

    # Compute the 10 smallest eigenvalues and corresponding eigenvectors
    num_eigenvalues = 10
    eigenvalues, eigenvectors = eigsh(sparse_matrix, k=num_eigenvalues, which='SM')

    # Print the 10 smallest eigenvalues
    print("10 smallest eigenvalues:")
    print(eigenvalues)


    # distinguish between mobile/imobile particles
    r_vecs_initial = np.array([1.0* b.location for b in bodies]) #all blobs
    fix_index = 150#first index for rigid body
    r0 = 1.0*r_vecs_initial[fix_index::,:] #choose only immobile blobs


    # rotate only the roller with constant omega
    # blobs in structures do not rotate
    Omega =[]

    for index in range(num_particles):
        if index < fix_index:  
            Omega.append(9.0*2.0*np.pi) # specify omega as a vector
        else:
            Omega.append(0) 


    total_rej = 0
    for n in range(150):
        print(n)

        FT_calc = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies,
                          g=read.g,
                          repulsion_strength_firm=repulsion_strength_delta,
                          debye_length_firm=debye_length_delta,
                          firm_delta=firm_delta,
                          repulsion_strength_wall=read.repulsion_strength_wall,
                          debye_length_wall=read.debye_length_wall,
                          repulsion_strength=read.repulsion_strength,
                          debye_length=read.debye_length,
                          periodic_length=L,
                          omega=0,  # Omega ############## CHANGE ME TO ZERO FOR CONST OMEGA AND TO 'Omega' FOR CONST TORQUE
                          eta=eta,
                          fix_index=fix_index,
                          r0=r0,
                          a=a)

        # Update roller with const. omega but limit torque/ hard cutoff
        #Torque_Lim = 1.9904
        #Output_Vel = True
        #t0 = time.time()
        #reject_wall, reject_jump, Trap_vel_t = LSolv.Update_Bodies_Trap(
            #FT_calc, Omega=Omega, Out_Torque=Output_Vel, Cut_Torque=Torque_Lim)
        #dt1 = time.time() - t0

        #Update rollers with const. omega and no torque limitaion
        Output_Vel = True
        t0 = time.time()
        reject_wall, reject_jump, Trap_vel_t = LSolv.Update_Bodies_Trap(FT_calc,Omega=Omega,Out_Torque=Output_Vel)
        dt1 = time.time() - t0

        # Update rollers with const. torque (ALSO MAKE CHANGE ON LINE 169 in FT_calc)
        #Output_Vel = False
        #t0 = time.time()
        #reject_wall, reject_jump = LSolv.Update_Bodies_Trap(FT_calc)
        #dt1 = time.time() - t0

        print(("walltime for time step : %s" % dt1))
        print(("Number of rejected timesteps wall: %s" %
              LSolv.num_rejections_wall))
        print(("Number of rejected timesteps jump: %s" %
              LSolv.num_rejections_jump))
