# AdePy

AdePy contains analytical solutions for the advection-dispersion equation (ADE) describing solute transport in groundwater, written in Python. 

Currently, all solutions shown in [Wexler (1992)](https://doi.org/10.3133/twri03B7) are provided as separate Python functions. These simulate 1D, 2D or 3D solute transport in uniform background flow for a variety of boundary conditions and source geometries. The solute may be subjected to linear sorption and first-order decay. 

Since all equations are linear, superposition in time and space can be applied to create complex source geometries with time-varying source concentrations. Gauss-Legendre quadrature is used to solve the integrals which require numerical integration.

## To install
Download or `git clone` the [GitHub repository](https://github.com/cneyens/adepy) locally. Then install using:

```
pip install -e <path/to/local/clone>
```

AdePy depends on [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Numba](https://numba.pydata.org/).

## Documentation
Coming soon.

## Available solutions

| Python function  | Dimensionality | Source geometry         | Boundary type | Aquifer geometry        | Reference     |
| -----------------|----------------|-------------------------|---------------|-------------------------|---------------|
| `finite1()`      | 1D             | Inlet                   | Dirichlet     | Finite                  | Wexler (1992) |
| `finite3()`      | 1D             | Inlet                   | Cauchy        | Finite                  | Wexler (1992) |
| `seminf1()`      | 1D             | Inlet                   | Dirichlet     | Semi-infinite           | Wexler (1992) |
| `seminf3()`      | 1D             | Inlet                   | Cauchy        | Semi-infinite           | Wexler (1992) |
|                  |                |                         |               |                         |               |
| `point2()`       | 2D             | Point                   | Cauchy        | Infinite                | Wexler (1992) |
| `stripf()`       | 2D             | Finite Y at X=0         | Dirichlet     | Finite Y                | Wexler (1992) |
| `stripi()`       | 2D             | Finite Y at X=0         | Dirichlet     | Semi-infinite           | Wexler (1992) |
| `gauss()`        | 2D             | Gaussian along Y at X=0 | Dirichlet     | Semi-infinite           | Wexler (1992) |
|                  |                |                         |               |                         |               |
| `point3()`       | 3D             | Point                   | Cauchy        | Infinite                | Wexler (1992) |
| `patchf()`       | 3D             | Finite Y and Z at X=0   | Dirichlet     | Finite Y and Z          | Wexler (1992) |
| `patchi()`       | 3D             | Finite Y and Z at X=0   | Dirichlet     | Semi-infinite           | Wexler (1992) |

## Example
The fate of a contaminant plume generated by continuous injection of a point source in an aquifer with uniform background flow is simulated. The source generates a plume which extends in three dimensions and migrates due to advection and mechanical dispersion. Molecular diffusion, linear sorption and first-order decay are neglected in this example.

```python
import numpy as np
import matplotlib.pyplot as plt
from adepy import point3 # 3D ADE solution of a continuous point source

# Source parameters ----
xc = 0   # x-coordinate of point source, m
yc = 0   # y-coordinate of point source, m
zc = 0   # z-coordinate of point source, m
c0 = 100 # injection concentration, mg/L
Q = 1    # injection rate, m^3/d

# Aquifer parameters ----
v = 0.05  # uniform groundwater flow velocity in x-direction, m/d
al = 5    # longitudinal dispersivity, m
ah = 1    # horizontal transverse dispersivity, m
av = 0.1  # vertical transverse dispersivity, m
n = 0.2   # porosity, -

# Calculate and plot the concentration field after 1 year at z = 0 ----
t = 365   # output time, d
x, y = np.meshgrid(np.linspace(-10, 20, 100), 
                   np.linspace(-7.5, 7.5, 100))  # output grid x-y coordinates, m
z = 0     # output grid z-coordinate, m

c = point3(c0, x, y, z, t, v, n, al, ah, av, Q, xc, yc, zc) # simulated concentration, mg/L

plt.contour(x, y, c, levels=np.arange(100, 2501, 100))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_aspect("equal")
plt.grid()
```
<img src="doc/_readme/readme_contours.png" width="75%" />

```python
# Calculate and plot the concentration time series for 5 years at a location downstream ----
obs = (10, 0, 0)  # x-y-z coordinates of observation point, m
t = np.linspace(1, 5 * 365, 100)  # output times, d
cobs = point3(c0, obs[0], obs[1], obs[2], t, v, n, al, ah, av, Q, xc, yc, zc)

plt.plot(t, cobs)
plt.xlabel('Time (d)')
plt.ylabel('Concentration (mg/L)')
```

<img src="doc/_readme/readme_ts.png" width="60%" />

## References
[Wexler, E.J., 1992. *Analytical solutions for one-, two-, and three-dimensional solute transport in ground-water systems with uniform flow*, USGS Techniques of Water-Resources Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7](https://doi.org/10.3133/twri03B7)