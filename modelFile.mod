
option solver cplex;
option show_stats 1;
option cplex_options 'baropt bardisplay=1 barstart=1 comptol=1e-8 crossover=0';

#instance parameter
param I; #number of resources
param L; #number of customer types
param q {i in 1..I}; #probabilities of finished jobs
param lambda {l in 0..L}; #probabilities for customer types
param r {l in 0..L}; #rewards
param beta; #cost for queue lengths
param alpha; #discount factor
param p {i in 1..I}; #probabilities of queue choice
param xmax; #maximal queue length
param Xmax := sum{i in 1..I}xmax*(xmax+1)^(i-1); #vector x as scalar X
param Amax := sum{l in 1..L}2^(l-1); #vector a as scalar A
param T; #large time horizon, in theory equal to infinity

set METHODS;
set METHODSLP := METHODS diff {'acceptAll','exact'}; #only ALP methods

#for exact value function
param vExact {X in 0..Xmax}; #exact value function
param vUpdate {X in 0..Xmax}; #for value iteration

param conindex;

#for constraint sampling
set SAMCONS {VFA in METHODSLP}; #set of constraints
param found;
param isequal;
param nrOfIt; #number of iterations
param sampleSize; #number of samples for constraint sampling
param Tsample; #time horizon for constraint sampling
param xhatSample {VFA in METHODSLP, cc in SAMCONS[VFA], i in 1..I};
param ahatSample {VFA in METHODSLP, cc in SAMCONS[VFA], l in 1..L};
param xRand {i in 1..I}; #random initial state x

#for constraint generation
set GENCONS {VFA in METHODSLP}; #set of constraints
param deltaTol; #tolerance parameter 
param epsilonErr; #constraint violation
param xhatGen {VFA in METHODSLP, cc in GENCONS[VFA], i in 1..I};
param ahatGen {VFA in METHODSLP, cc in GENCONS[VFA], l in 1..L};
param oldMax1;
param newMax1;
param oldMax2;
param newMax2;
param currXmax {i in 1..I};

#simulation
param xSim {VFA in METHODS, i in 1..I}; #state vector x
param xAverage {VFA in METHODS, i in 1..I, x in 0..xmax};
param nrOfSim; #number of simulations
param reward {VFA in METHODS, s in 1..nrOfSim}; #cumulated reward
param averageReward {VFA in METHODS};
param confidence {VFA in METHODS}; 
param xStart {i in 1..I}; #initial state x
param XStart := sum{i in 1..I} xStart[i]*(xmax+1)^(i-1); #initial state vector as scalar
param dice; #for random processes
param Astar {VFA in METHODS}; #maximizing action as scalar
param alstar {VFA in METHODS}; #maximizing action a_l
param decision {VFA in METHODS, A in 0..Amax}; 
param auxMax;
param i1star;
param i2star;
param lstar;

#value function approximation for simulation
param VAFsim {i in 1..I};
param thetaAFsim;
param vSPLsim {i in 1..I, x in 0..xmax};

#---------------------------------------------------------

var VAF {i in 1..I};
var thetaAF;
var zAF {i in 1..I, A in 0..Amax};
var vSPL {i in 1..I, x in 0..xmax};
var zSPL {i in 1..I, A in 0..Amax};

minimize ZSPL: sum{i in 1..I, x in 0..xmax} 1/(xmax+1)*vSPL[i,x];
subject to conSPL {A in 0..Amax, X in 0..Xmax}:
	sum{i in 1..I} vSPL[i,floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))] 
		>= (-beta*sum{i in 1..I}floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
			+sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*(
				(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*r[l]*(if floor((X mod (xmax+1)^i1)/((xmax+1)^(i1-1))) < xmax then 1 else 0)
				+alpha*sum{i in 1..I}vSPL[i,
					max(0,floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
						+(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*(if i=i1 and floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))<xmax then 1 else 0)
						-(if i=i2 then 1 else 0))
				]
			)
		);
subject to conSPLred1 {A in 0..Amax}: 0 >= sum{i in 1..I} zSPL[i,A];
subject to conSPLred2 {i in 1..I, x in 0..xmax, A in 0..Amax}:
	zSPL[i,A] >= -vSPL[i,x] - beta*x 
	+sum{l in 0..L} lambda[l]*p[i]*(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*r[l]*(if x<xmax then 1 else 0)
	+alpha*sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*vSPL[i,max(0,x
		+(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*(if i=i1 and x<xmax then 1 else 0)
		-(if i=i2 then 1 else 0))];

minimize ZAF: thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax;
subject to conAF {A in 0..Amax, X in 0..Xmax}:
	thetaAF+sum{i in 1..I} VAF[i]*floor((X mod (xmax+1)^i)/((xmax+1)^(i-1))) 
		>= (-beta*sum{i in 1..I}floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
			+sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*(
				(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*r[l]*(if floor((X mod (xmax+1)^i1)/((xmax+1)^(i1-1))) < xmax then 1 else 0)
				+alpha*(thetaAF+sum{i in 1..I}VAF[i]*
					max(0,floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
						+(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*(if i=i1 and floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))<xmax then 1 else 0)
						-(if i=i2 then 1 else 0))
				)
			)
		);

subject to conAFsample {cc in SAMCONS['AF']}:
	(1-alpha)*thetaAF 
	>= sum{i in 1..I} (
		-VAF[i]*xhatSample['AF',cc,i] - beta*xhatSample['AF',cc,i]
		+sum{l in 0..L}lambda[l]*p[i]*(if l=0 then 0 else ahatSample['AF',cc,l])*r[l]*(if xhatSample['AF',cc,i] < xmax then 1 else 0)
		+alpha*sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*VAF[i]*
			max(0,xhatSample['AF',cc,i]
				+(if l=0 then 0 else ahatSample['AF',cc,l])*(if i=i1 and xhatSample['AF',cc,i]<xmax then 1 else 0)
				-(if i=i2 then 1 else 0))
	);
	
subject to conAFgen {cc in GENCONS['AF']}:
	(1-alpha)*thetaAF 
	>= sum{i in 1..I} (
		-VAF[i]*xhatGen['AF',cc,i] - beta*xhatGen['AF',cc,i]
		+sum{l in 0..L}lambda[l]*p[i]*(if l=0 then 0 else ahatGen['AF',cc,l])*r[l]*(if xhatGen['AF',cc,i] < xmax then 1 else 0)
		+alpha*sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*VAF[i]*
			max(0,xhatGen['AF',cc,i]
				+(if l=0 then 0 else ahatGen['AF',cc,l])*(if i=i1 and xhatGen['AF',cc,i]<xmax then 1 else 0)
				-(if i=i2 then 1 else 0))
	);
	
subject to conAFred1 {A in 0..Amax}: (1-alpha)*thetaAF >= sum{i in 1..I} zAF[i,A];
subject to conAFred2 {i in 1..I, x in 0..xmax, A in 0..Amax}:
	zAF[i,A] >= -VAF[i]*x - beta*x 
	+sum{l in 0..L} lambda[l]*p[i]*(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*r[l]*(if x<xmax then 1 else 0)
	+alpha*sum{l in 0..L, i1 in 1..I, i2 in 1..I}lambda[l]*p[i1]*q[i2]*VAF[i]*max(0,x
		+(if l=0 then 0 else floor((A mod 2^l)/(2^(l-1))))*(if i=i1 and x<xmax then 1 else 0)
		-(if i=i2 then 1 else 0));

subject to conAFadditional:
	thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax
	>= -beta*I*xmax/(1-alpha);

problem SPL: ZSPL, vSPL, conSPL;
problem SPLred: ZSPL, vSPL, zSPL, conSPLred1, conSPLred2;
problem AF: ZAF, thetaAF, VAF, conAF;
problem AFsample: ZAF, VAF, thetaAF, conAFsample, conAFadditional;
problem AFgen: ZAF, VAF, thetaAF, conAFgen, conAFadditional;
problem AFred: ZAF, VAF, thetaAF, zAF, conAFred1, conAFred2;
