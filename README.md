Download Link: https://assignmentchef.com/product/solved-hpc-project-4-parallel-space-solution-of-a-nonlinear-pde-using-openmp
<br>



<h1>The Fisher’s equation as an example of a reaction-diffusion PDE</h1>

The simple OpenMP exercises such as the <em>π</em>-computation or the parallel Mandelbrot set are good examples for the basic understanding of OpenMP constructs, but due to their simplicity they are far away from practical applications. In the HPC Lab for CSE we are going to use a parallel PDE mini-application that solves something more sophisticated, but where the code is nevertheless still relatively short and easy to understand. Our mini-application will solve a prototype of a reaction-diffusion equation (1). It is the Fisher’s Equation that can be used to simulate travelling waves and simple population dynamics, and is given by

in Ω<em>,                                                                 </em>(1)

where <em>s </em>:= <em>s</em>(<em>x,y,t</em>), <em>D </em>is the diffusion constant and <em>R </em>is the reaction constant. The left hand side represents the rate of change of <em>s </em>with time. On the right hand side, the first term describes the diffusion of <em>s </em>in space and the second term describes the reaction or growth of the wave or population. We set the domain to be the unit square, i.e. Ω = (0<em>,</em>1)<sup>2</sup>, and prescribe Dirichlet boundary conditions on the whole boundary <em>∂</em>Ω with a fixed constant value of zero. The initial conditions <em>s</em><sup>init </sup>are set to zero over the entire domain, except for a circular region in the bottom left corner that is initialized to the value 0<em>.</em>1. The domain Ω is discretized using a uniform grid with (<em>n </em>+ 2)×(<em>n </em>+ 2) points, where a grid point is indicated by <em>x<sub>i,j</sub></em>, for <em>i,j </em>∈ {0<em>,</em>1<em>,…,n </em>+ 1}, and where <em>s<sup>k</sup><sub>i,j </sub></em>is the approximation of <em>s </em>at the grid-point <em>x<sub>i,j </sub></em>at time-step <em>k</em>, see Figure 1.

We use a second-order finite difference discretization to approximate the spatial derivatives of <em>s </em>for all inner grid points for a fixed time <em>t</em>,

<em>,                                        </em>(2)

for all (<em>i,j</em>) ∈ {1<em>,…,n</em>}, and where ∆<em>x </em>= 1<em>/</em>(<em>n </em>+ 1), for ∆<em>x </em>sufficiently small. In order to approximate the time derivative, we use a first-order finite difference scheme for each grid point <em>x<sub>i,j</sub></em>, which at time-step <em>k </em>gives

<em>,                                                                                 </em>(3)

where ∆<em>t </em>is the time-step size. Putting together the components above, we obtain the following discretization of Equation (1):

<em>,                      </em>(4)

<table width="680">

 <tbody>

  <tr>

   <td width="663">that we will attempt to solve in order to obtain an approximate solution for <em>s</em>. We can reformulate (4) as</td>

   <td width="17"></td>

  </tr>

  <tr>

   <td width="663"><em>f</em><em>i,jk </em>:= [−(4 + <em>α</em>)<em>s</em><em>i,j </em>+ <em>s</em><em>i</em>−1<em>,j </em>+ <em>s</em><em>i</em>+1<em>,j </em>+ <em>s</em><em>i,j</em>−1 + <em>s</em><em>i,j</em>+1 + <em>βs</em><em>i,j</em>(1 − <em>s</em><em>i,j</em>)]<em>k </em>+ <em>αs</em><em>ki,j</em>−1 = 0</td>

   <td width="17">(5)</td>

  </tr>

 </tbody>

</table>

for each tuple (<em>i,j</em>) and time-step <em>k </em>with <em>α </em>:= ∆<em>x</em><sup>2</sup><em>/</em>(<em>D</em>∆<em>t</em>) and <em>β </em>:= <em>R</em>∆<em>x</em><sup>2</sup><em>/D</em>. At time-step <em>k </em>= 1, we initialize , and we look for approximate values <em>s<sup>k</sup><sub>i,j </sub></em>that fulfill Equation (5). For all (<em>i,j</em>) at a fixed <em>k</em>,

Figure 1: Discretization of the domain Ω.

we obtain a system of <em>N </em>= <em>n</em><sup>2 </sup>equations. We can see that each equation is quadratic in <em>s<sup>k</sup><sub>i,j</sub></em>, and therefore nonlinear which makes the problem more complicated to solve. We tackle this problem using Newton’s method with which we iteratively try to find better approximations of the solution of Equation (5). In order to formulate the Newton iteration we introduce the following notation: Let <strong>s</strong>be a vectorized version of the approximate solution at time-step <em>k</em>. Then, we can consider the set of equations <em>f<sub>i,j</sub><sup>k </sup></em>as functions depending on <strong>s</strong><em><sup>k </sup></em>and define <strong>f</strong>. For Newton’s method, we then have

<strong>y</strong><em><sup>l</sup></em><sup>+1 </sup>= <strong>y</strong><em><sup>l </sup></em>− [<strong>J<sub>f</sub></strong>(<strong>y</strong><em><sup>l</sup></em>)]−<sup>1</sup><strong>f</strong>(<strong>y</strong><em><sup>l</sup></em>)<em>,                                                                                    </em>(6)

where <strong>J<sub>f</sub></strong>(<strong>y</strong><em><sup>l</sup></em>) ∈ R<em><sup>N</sup></em>×<em><sup>N </sup></em>is the Jacobian of <strong>f</strong>. We start with the initial guess <strong>y</strong><sup>0 </sup>:= <strong>s</strong><em><sup>k</sup></em>−<sup>1</sup>. However, for each iteration the inverse of the Jacobian [<strong>J<sub>f</sub></strong>(<strong>y</strong><em><sup>l</sup></em>)]−<sup>1 </sup>is not readily available. We do not compute it directly but instead use a matrix-free Conjugate Gradient solver that solves the following linear system of equations for <em>δ</em><strong>y</strong><em><sup>l</sup></em><sup>+1</sup>:

[<strong>J<sub>f</sub></strong>(<strong>y</strong><em><sup>l</sup></em>)]<em>δ</em><strong>y</strong><em><sup>l</sup></em><sup>+1 </sup>= <strong>f</strong>(<strong>y</strong><em><sup>l</sup></em>)                                                                                         (7)

and for which it follows that <strong>y</strong>. We iterate over <em>l </em>in Equation (6) till a stopping criterion is reached and we obtain a final solution <strong>y </strong>. This is the approximate solution for time-step <em>k</em>, i.e. <strong>s</strong><em><sup>k </sup></em>:= <strong>y</strong><sup>fin </sup>that we originally set out to find in Equation (5). It is then in turn used as the initial guess to compute an approximate solution for the next time-step <em>k </em>+ 1.

<h1>Code Walkthrough</h1>

The provided code for the project already contains most of the functionalities described above, a brief overview of the code is presented in the this section. The main task of this project will be (i) to complete some parts of the sequential code and (ii) the parallelization of the code using OpenMP. This project will also serve as an example for using the message-passing interface MPI in project 4. There are three files of interest:

<ul>

 <li>cpp: initialization and main time-stepping loop</li>

 <li>cpp: the BLAS level 1 (vector-vector) kernels and conjugate gradient solver

  <ul>

   <li>Interior grid points</li>

   <li>Boundary grid points</li>

  </ul></li>

</ul>

Figure 2: Visualization of stencil operators. The blue grid points indicate the grid points on the boundary of the domain with fixed values (see right side, set to zero in our setting). The orange grid points indicate inner grid points of the domain, whose stencil does not depend on the boundary values. The green grid points are still inside the domain but their stencils require boundary values.

<ul>

 <li>This file defines simple kernels for operating on 1D vectors, including</li>

</ul>

∗ dot product: <em>x </em>· <em>y</em>: hpcdot()

∗ linear combination: <em>z </em>= alpha ∗ <em>x </em>+ beta ∗ <em>y</em>: hpclcomb() ∗ …

<ul>

 <li>All the kernels of interest in this HPC Lab for CSE start with hpcxxxxx</li>

</ul>

<ul>

 <li>cpp: the stencil operator for the finite difference discretization

  <ul>

   <li>This file has a function/subroutine that defines the stencil operator</li>

   <li>The stencil operators differ depending on the position of the grid point, see Figures (2) and Algorithms 1 and 2:</li>

  </ul></li>

</ul>

for <em>j </em>= 2 : <em>ydim </em>− 1 do for <em>i </em>= 2 : <em>xdim </em>− 1 do

<em>f</em><em>i,j </em>= [−(4 + <em>α</em>)<em>s</em><em>i,j </em>+ <em>s</em><em>i</em>−1<em>,j </em>+ <em>s</em><em>i</em>+1<em>,j </em>+ <em>s</em><em>i,j</em>−1 + <em>s</em><em>i,j</em>+1 + <em>βs</em><em>i,j</em>(1 − <em>s</em><em>i,j</em>)]<em>k</em>+1 + <em>αs</em><em>ki,j </em>= 0

end end

Algorithm 1: Stencil: interior grid points

<em>i </em>= <em>xdim</em>

for <em>j </em>= 2 : <em>ydim </em>− 1 do

end

Algorithm 2: Stencil: boundary grid points

<h1>Compile and run the PDE mini-app on the ICS cluster</h1>

Log-in to the ICS cluster and afterwards load the gcc and python modules.

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="5c292f392e1c30333b3532">[email protected]</a>]$ module load gcc python

Go to the Project4 directory and use the makefile to compile the code

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="295c5a4c5b6945464e4047">[email protected]</a>]$ make

Run the application on a compute node with selected parameters, e.g. domain size 128 × 128, 100 time steps and simulation time 0 − 0<em>.</em>01s, for now using a single OpenMP thread.

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="5326203621133f3c343a3d">[email protected]</a>]$ salloc

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="d4a1a7b1a694bdb7a7babbb0b18c8c">[email protected]</a>]$ export OMP_NUM_THREADS=1

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="0174726473416862726f6e65645959">[email protected]</a>]$ ./main 128 100 0.01

After you implement the first part of the assignment, the output of the mini-application should look like this:

<table width="675">

 <tbody>

  <tr>

   <td width="675">======================================================================== Welcome to mini-stencil! version :: C++ Serial mesh           :: 128 * 128 dx = 0.00787402 time         :: 100 time steps from 0 .. 0.01iteration :: CG 200, Newton 50, tolerance 1e-06======================================================================== ——————————————————————————-simulation took 1.06824 seconds7703 conjugate gradient iterations, at rate of 7210.92 iters/second866 newton iterations——————————————————————————-Goodbye!</td>

  </tr>

 </tbody>

</table>

<h1>1.      Task: Implementing the linear algebra functions and the stenciloperators [40 Points]</h1>

The provided implementation is a serial version of the PDE mini-application with some code missing. Your first task is to implement the missing code to get a working PDE mini-application.

<ul>

 <li>Open the file linalg.cpp and implement the functions hpcxxxxx(). Follow the comments in the code as they are there to help you with the implementation.</li>

 <li>Open file operators.cpp and implement the missing stencil kernels.</li>

</ul>

After completion of the above steps, the mini-app should produce correct results. Compare the number of conjugate gradient iterations and the Newton iterations with the reference output above. If the numbers are about the same, you have probably implemented everything correctly. Now try to plot the solution with the script plotting.py. It should look like in Figure 3.

$ ./plotting.py

<h2>Note</h2>

The script plotting.py reads the results computed by ./main and creates an image output.png.

If X11 forwarding is enabled, it can also show the image in a window. To enable X11 forwarding, connect to ICS cluster with parameter -Y

Figure 3: Output of the mini-app for a domain discretization into 128×128 grid points, 100 time steps with a simulation time from 0 − 0<em>.</em>01s

$ ssh -Y icsmaster

Then allocate a compute node with X11 forwarding and execute the plotting script on the compute node

$ salloc –x11

Note that to use the X11 forwarding, you need X server installed on your computer. On most Linux distributions it is already installed. On MacOS you have to download and install <a href="https://www.xquartz.org/">XQuartz</a> and on Windows you have to download and install <a href="https://sourceforge.net/projects/xming/">Xming.</a> After the installation, you might need to reboot your computer.

If the plotting script prints an error

No module named ’matplotlib’

you probably forgot to load the python module. Please call

$ module load python

<h1>2.     Task: Adding OpenMP to the nonlinear PDE mini-app [60 Points]</h1>

When the serial version of the mini-app is working we can add OpenMP directives. This allows you to use all cores on one compute node. In this project 3, you will measure and improve the scalability of your PDE simulation code. Scalability is the changing performance of a parallel program as it utilizes more computing resources. In this project we are interested in a performance analysis of both strong scalability and weak scalability. Strong scaling is identifying how a threaded PDE solver gets faster for a fixed PDE problem size. That is, we have the same discretization points per direction, but we run it with more and more threads and we hope/expect that it will compute its result faster. Weak scaling speaks to the latter point: how effectively can we increase the size of the problem we are dealing with? In a weak scaling study, each compute core has the same amount of work, which means that running on more threads increases the total size of the PDE simulation.

<h2>Replace welcome message in main.cpp</h2>

Replace the welcome message in main.cpp with a message that informs the user about the following points:

<ul>

 <li>That this is the OpenMP version.</li>

 <li>The number of threads it is using.</li>

</ul>

For example:

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="4134322433012d2e26282f">[email protected]</a>]$ salloc

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="295c5a4c5b69404a5a47464d4c7171">[email protected]</a>]$ export OMP_NUM_THREADS=8

[<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="2154524453614842524f4e45447979">[email protected]</a>]$ ./main 128 100 0.01

======================================================================== Welcome to mini-stencil! version :: C++ OpenMP threads :: 8 …

<h2>Linear algebra kernel</h2>

Open linalg.cpp and add OpenMP directives to all functions hpcXXXX(), except for hpccg().

<ul>

 <li>Recompile frequently and run with 8 threads to check that you are getting the right answer.</li>

</ul>

Once finished with this file, did your changes make any performance improvement?

<ul>

 <li>Compare the 128×128 and 256×256</li>

</ul>

<h2>The diffusion stencil</h2>

The final step is to parallelize the stencil operator in operators.cpp,

<ul>

 <li>The nested for loop is an obvious target.</li>

 <li>It covers inner grid points.</li>

 <li>How about the boundary loops?</li>

</ul>

<h2>2.1.      Strong scaling</h2>

Before starting adding OpenMP parallelism, find two sets of input parameters in term of larger grid size that converge for the serial version.

<ul>

 <li>You can use these parameters, or choose anything you like,

  <ul>

   <li>./main 256 100 0.01</li>

   <li>./main 512 100 0.01</li>

  </ul></li>

 <li>Write down the time to solution.

  <ul>

   <li>You want this to get faster as you add OpenMP.</li>

   <li>But you might have to add a few additional directives (such as SIMD instruction) before things actually get faster. Write down the number of conjugate gradient iterations.</li>

   <li>Use this to check after you add each directive that you are still getting the right answer.</li>

   <li>Remember that there will be some small variations because floating point operations are not commutative.</li>

   <li>Review your OpenMP code and argue if a threaded OpenMP PDE solver can be implemented which produces bitwise-identical results without any parallel side effects.</li>

  </ul></li>

</ul>

How does it scale at different resolutions? Plot time to time to solution for the grid sizes below for 1-10 threads .

<ul>

 <li>64×64</li>

 <li>128×128 • 256×256</li>

 <li>512×512</li>

 <li>1024×1024</li>

</ul>

<h2>2.2.      Weak scaling</h2>

Produce a weak scaling plot by running the PDE code with different numbers of threads and a fixed constant grid size per thread. When performing this weak-scaling study, we are asking a complementary question to that asking in a strong-scaling study. Instead of keeping the problem size fixed, we increase the problem size relative to the number of cores. For example, we might initially run a PDE simulation on a grid of 64×64 using 4 cores. Later, we might want to compare 128×128 using 16 cores. Will the time remain constant, since the work per core has remained the same, or will the time increase due, for example, to increased communications overhead associated with the larger problem size, or number of cores, or a higher number of nonlinear iterations? A weak scaling study is one method of documenting this behavior for your application, allowing you to make an guess at the amount of computing time you need.

<ul>

 <li></li>

</ul>