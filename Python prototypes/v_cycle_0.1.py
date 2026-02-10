# Multigrid V-cycle implementation in Python
# This code implements a simple multigrid method for solving Poisson's equation
# using a V-cycle approach. It includes relaxation, restriction, and prolongation steps.



# we'll have our region of interest be [0, 1] x [0, 1]
# and we will discretize it into a grid of size n x n
# and use a grid of size n x n

# There is a 40 degree celsius heater in [0, 1/4] x [0, 1/4]

# n = 128  # Grid size
# temp = np.full((n, n), 20) # Initial temperature
# temp[:n//4, :n//4] = 40.0  # Set heater temperature in the top-left corner
# temp[:, 0] = 0.0  # Set left boundary condition
# temp[:, -1] = 0.0  # Set right boundary condition
# temp[0, :] = 0.0  # Set bottom boundary condition
# temp[-1, :] = 0.0  # Set top boundary condition

# plt.imshow(temp, cmap='viridis', interpolation='nearest')  # Plot initial condition
# plt.show()

# def jacobi_step(temp, n):
#     """Perform one Jacobi iteration."""
#     new_temp = np.copy(temp)
#     for i in range(1, n-1):
#         for j in range(1, n-1):
#             new_temp[i, j] = 0.25 * (temp[i-1, j] + temp[i+1, j] + temp[i, j-1] + temp[i, j+1]) # average of the four neighbors
#     new_temp[:n//4, :n//4] = 40.0  # Reapply heater temperature in the top-left corner
#     new_temp[0, :] = 0.0  # Enforce bottom boundary condition
#     new_temp[-1, :] = 0.0  # Enforce top boundary condition
#     new_temp[:, 0] = 0.0  # Enforce left boundary condition
#     new_temp[:, -1] = 0.0  # Enforce right boundary condition
#     return new_temp

# def jacobi_method(temp, n, iterations=1000):
#     """Perform Jacobi relaxation."""
#     for _ in range(iterations):
#         temp = jacobi_step(temp, n)
#     return temp

# solution = jacobi_method(temp, n, iterations=1000)


# # plt.imshow(solution, cmap='viridis', interpolation='nearest')  # Plot solution
# # plt.colorbar(label='Temperature (Celsius)')
# # plt.show()

# # """
# # - Create a flattening function to convert the grid to a 1D array
# # - Create an unflattening function to convert a 1D array back to the grid
# # - Create a restriction function to downsample the grid
# # - Create an interpolation function to upsample the grid
# # """

# def flatten(grid):
#     """Flatten a 2D grid to a 1D array."""
#     return grid.flatten()

# def unflatten(array, n):
#     """Unflatten a 1D array back to a 2D grid."""
#     return array.reshape((n, n))

# def restrict(grid):
#     """Restrict grid to a coarser grid."""
#     
#     n = grid.shape[0]
#     coarse = np.zeros((n//2 + 1, n//2 + 1))
#     for i in range(0, n//2 + 1):
#         for j in range(0, n//2 + 1):
#             coarse[i, j] = grid[2*i, 2*j]
#     # resetting the boundaries of the coarse grid
#     coarse[0: n//4, 0: n//4] = 40.0  # Reapply heater temperature in the top-left corner
#     coarse[0, :] = 0.0  # Enforce bottom boundary condition
#     coarse[-1, :] = 0.0  # Enforce top boundary condition   
#     coarse[:, 0] = 0.0  # Enforce left boundary condition
#     coarse[:, -1] = 0.0  # Enforce right boundary condition
    
#     
#     return coarse

def interpolate(coarse):
    """Interpolate coarse grid to a finer grid."""
    n = coarse.shape[0]
    fine = np.zeros((2*n, 2*n))
    for i in range(n):
        for j in range(n):
            fine[2*i, 2*j] = coarse[i, j]
            if 2*i+1 < fine.shape[0]:
                fine[2*i+1, 2*j] = 0.5 * (coarse[i, j] + coarse[min(i+1, n-1), j])
            if 2*j+1 < fine.shape[1]:
                fine[2*i, 2*j+1] = 0.5 * (coarse[i, j] + coarse[i, min(j+1, n-1)])
            if 2*i+1 < fine.shape[0] and 2*j+1 < fine.shape[1]:
                fine[2*i+1, 2*j+1] = 0.25 * (coarse[i, j] + coarse[min(i+1, n-1), j] + coarse[i, min(j+1, n-1)] + coarse[min(i+1, n-1), min(j+1, n-1)])
    return fine

# my_coarse = restrict(temp)

# my_fine = interpolate(my_coarse)

# # Plot the coarse grid
# plt.imshow(my_coarse, cmap='viridis', interpolation='nearest')  # Plot the coarse grid 
# plt.show()

# plt.imshow(my_fine, cmap='viridis', interpolation='nearest')  # Plot the fine grid
# plt.show()

# def multigrid_v_cycle(temp, n, levels=3, iterations=5):

"""
Here's my understanding of the multigrid V-cycle algorithm:
- Start with your finest grid with your initial and boundary conditions
- Perform pre-smoothing on the grid using a relaxation method (e.g., Jacobi or Gauss-Seidel)
"""
import numpy as np
import matplotlib.pyplot as plt

# recall that we don't want to solve anything if we're on the boundaries or in the heater region
def in_heater(i, j, N):
    return i <= N/4 + 1 and j <= N/4 +1

def on_boundary(i, j, N):
    return i==0 or j==0 or i==N or j==N

def relax(grid, rhs, omega=0.6, iterations=5):
    """Perform Gauss-Seidel relaxation."""
    n = grid.shape[0]
    for _ in range(iterations):
        for i in range(1, n-1):
            for j in range(1, n-1):
                if in_heater(i, j, n-1) or on_boundary(i, j, n-1):
                    continue
                else:
                    grid[i, j] = (1 - omega) * grid[i, j] + omega * 0.25 * (
                                grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + 
                                grid[i, j+1] + rhs[i, j])
    # Enforce boundary conditions
    grid[0:n//4, 0:n//4] = 40.0  # Reapply heater temperature in the top-left corner
    grid[0, :] = 0.0  # Enforce bottom boundary condition
    grid[-1, :] = 0.0  # Enforce top boundary condition
    grid[:, 0] = 0.0  # Enforce left boundary condition
    grid[:, -1] = 0.0  # Enforce right boundary condition
    

def restrict(grid):
    """Restrict grid to a coarser grid."""
    # plt.imshow(grid, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Temperature (Celsius)')
    # plt.title('before restriction')
    # plt.show()
    n = grid.shape[0]
    coarse = np.zeros((n//2 + 1, n//2 + 1))
    for i in range(1, coarse.shape[0]-1):
        for j in range(1, coarse.shape[1]-1):
            fi, fj = 2 * i, 2 * j
            # 9-point stencil
            coarse[i, j] = (
                1/16 * grid[fi-1, fj-1] + 1/8 * grid[fi-1, fj] + 1/16 * grid[fi-1, fj+1] +
                1/8  * grid[fi,   fj-1] + 1/4 * grid[fi,   fj] + 1/8  * grid[fi,   fj+1] +
                1/16 * grid[fi+1, fj-1] + 1/8 * grid[fi+1, fj] + 1/16 * grid[fi+1, fj+1]
            )

            # 4-point stencil
            # coarse[i, j] = 0.25 * (grid[2*i, 2*j] + grid[2*i-1, 2*j] + grid[2*i, 2*j-1] + grid[2*i-1, 2*j-1])

            # injection (copying and pasting the same values, no smoothing / stenciling)
            # coarse[i, j] = grid[fi, fj]
    print(f"restriction step from {grid.shape} to {coarse.shape}")
    print("* * * * * * * * * * *")

    # plt.imshow(coarse, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Temperature (Celsius)')
    # plt.title('after restriction')
    # plt.show()
    return coarse

def prolong(coarse):
    """Prolong coarse grid to a finer grid."""
    
    n = coarse.shape[0]
    fine = np.zeros((2*n - 1, 2*n - 1))
    for i in range(n - 1):  # avoid last row/col
        for j in range(n - 1):
            fi, fj = 2*i, 2*j

            # Copy coarse grid point to fine grid
            fine[fi, fj] = coarse[i, j]

            # Interpolate in x-direction (vertical midpoints)
            fine[fi+1, fj] = 0.5 * (coarse[i, j] + coarse[i+1, j])

            # Interpolate in y-direction (horizontal midpoints)
            fine[fi, fj+1] = 0.5 * (coarse[i, j] + coarse[i, j+1])

            # Interpolate center point (between 4 coarse grid points)
            fine[fi+1, fj+1] = 0.25 * (coarse[i, j] + coarse[i+1, j] +
                                       coarse[i, j+1] + coarse[i+1, j+1])
    print(f"prolongation step from {coarse.shape} to {fine.shape}")
    print("* * * * * * * * * * * * * *")
    
    

    return fine

def compute_residual(grid, rhs):
    n  = grid.shape[0] # dealing with a square grid
    residual = np.zeros_like(grid)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if in_heater(i, j, n-1) or on_boundary(i, j, n-1):
                continue
            else:
                laplace = -(4 * grid[i, j]
                        - grid[i-1, j]  # up
                        - grid[i+1, j]  # down
                        - grid[i, j-1]  # left
                        - grid[i, j+1]) # right

            residual[i, j] = rhs[i, j] - laplace

    return residual


def multigrid(grid, rhs, levels, omega=0.6, iterations=10):
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Temperature (Celsius)')
    plt.title(f'Initial Conditions at level {levels}')
    plt.show()

    """Multigrid V-cycle."""
    
    if levels == 1:
        relax(grid, rhs, omega, iterations)
        
        return grid

    # Pre-smoothing
    relax(grid, rhs, omega, iterations)

    # Compute residual
    # residual = rhs - (4 * grid[1:-1, 1:-1] - grid[:-2, 1:-1] - grid[2:, 1:-1] - grid[1:-1, :-2] - grid[1:-1, 2:])
    residual = compute_residual(grid, rhs)
    # residual = np.pad(residual, 1, mode='constant')

    # Restrict residual to coarser grid
    # print("the shape of the residual is", residual.shape)
    # print("the shape of the grid is", grid.shape)
    coarse_residual = restrict(residual)

    # Solve on coarser grid
    
    coarse_grid = np.zeros_like(coarse_residual)
    # coarse_grid = restrict(grid)
    correction = multigrid(coarse_grid, coarse_residual, levels - 1, omega, iterations)
    
    # Prolong correction to finer grid
    correction = prolong(correction)
    # correction[0:n//4, 0:n//4] = 0 # we don't need to add correction to the fixed heater zone

    # Apply correction
    print(type(grid))
    print(type(correction))
    grid = np.add(grid, correction)
    fine_level = compute_residual(grid, rhs)
    plt.imshow(fine_level, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='error')
    plt.title(f'Before relaxation residue (Level {levels})')
    plt.show()

    # Post-smoothing
    relax(grid, rhs, omega, iterations)
    relax(grid, rhs, omega, iterations)
    fine_level = compute_residual(grid, rhs)
    plt.imshow(fine_level, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='error')
    plt.title(f'After relaxation residue (Level {levels})')
    plt.show()

    return grid



#  document and print error at each level (once during restriction, once during interpolation)
"""
At each level going down the V-cycle, compute the L2 norm of the residual and store it / print it
"""


# Example usage
n = 129  # Grid size
levels = 6  # Multigrid levels
grid = np.full((n, n), 20)
grid[0:n//4, 0:n//4] = 40.0  # Heater in the top-left corner
grid[0, :] = 0.0  # Set bottom boundary condition
grid[-1, :] = 0.0  # Set top boundary condition
grid[:, 0] = 0.0  # Set left boundary condition
grid[:, -1] = 0.0  # Set right boundary condition

initial = grid.copy()
rhs = np.zeros((n, n))

solution = multigrid(grid, rhs, levels)
print(type(solution))

plt.imshow(initial, cmap='viridis', interpolation='nearest')  # Plot initial condition
plt.colorbar(label='Temperature (Celsius)')
plt.title('Initial Condition')
plt.show()

plt.imshow(solution, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Temperature (Celsius)')
plt.title('Multigrid Solution')
plt.show()