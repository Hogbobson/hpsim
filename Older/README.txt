v0.1.N
Programme doesn't work yet

v0.2.0
The programme works now, for the solar system consisting of 5 planets + sun.

v0.2.1
Added plotting, if poorly.

v0.2.2:
Change from 0.2.1: Added Saturn, Uranus, and Neptune. Cleaned up plotting a bit.

v0.2.3:
Cleaned up quite a lot, made everything more readable.
Acceleration function made more general, allowing for multiple forces.
Known exploitables - listing the same force key several times will let the force be calculated several times.

v0.2.4:
Minor optimization in the sym1 function.

v0.2.5:
Added symplectic solvers of 2nd, 3rd, and 4th order.
Version control protocol changed to saving version before trying to run it, as to not override previous working version.

v0.2.6:
Made an r_legacy variable to save all rs, easier for plotting, maybe expensive on time, we'll see.

v0.3.0:
Rewrote the program to largely be based on dictionaries instead of classes. 
Ensemble is now a dictionary, and functions called can be anything, as long as they output the proper things.
"The proper things" have yet to be properly described. Future work includes putting in a test to see if the ensemble is right.
The symplectic solvers of 3rd and 4th order are also broken, due to me being unable to make them to the standard I was capable of
before under the new system. This must change eventually.
The programme also doesn't plot anymore.

v0.3.1:
Added a function to randomly generate the solar system.

v0.3.2:
Re-added simple plotting. Todo: Look at what Brian sent's plotting routines. Steal them - I mean get inspired by them.
Known issues: the initial kick is wrong. Looking to fix it.
