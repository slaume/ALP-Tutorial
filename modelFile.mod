
option solver cplex;
option show_stats 1;
option cplex_options 'baropt bardisplay=1 barstart=1 comptol=1e-8 crossover=0';

#instance parameter
param I; #number of resources
param L; #number of customer types
param q {i in 1..I}; #probabilities of finished jobs
param lambda {l in 1..L}; #probabilities for customer types
param r {l in 1..L}; #rewards
param beta; #cost for queue lengths
param alpha; #discount factor
param p {i in 1..I}; #probabilities of queue choice
param xmax; #maximal queue length
param Kmax := sum{i in 1..I}2^(i-1); #vector k as scalar K
param Xmax := sum{i in 1..I}xmax*(xmax+1)^(i-1); #vector x as scalar X
param Amax := sum{l in 1..L}2^(l-1); #vector a as scalar A
param T; #large time horizon, in theory equal to infinity

set METHODS := {'AF','acceptAll'};
set METHODSLP := METHODS diff {'acceptAll'}; #only ALP methods

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
param istar;
param lstar;

#value function approximation for simulation
param VAFsim {i in 1..I};
param thetaAFsim;

#---------------------------------------------------------

var VAF {i in 1..I} <= 0;
var thetaAF;
var zAF {i in 1..I, A in 0..Amax};

minimize ZAF: thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax;
subject to conAFsample {cc in SAMCONS['AF']}:
	(1-alpha)*thetaAF >= sum{i in 1..I} (
		-VAF[i]*xhatSample['AF',cc,i]-beta*xhatSample['AF',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatSample['AF',cc,l]*r[l]*(if xhatSample['AF',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*VAF[i]*max(0,xhatSample['AF',cc,i]-1+ahatSample['AF',cc,l]*(if xhatSample['AF',cc,i] < xmax then 1 else 0))
			+(1-q[i])*VAF[i]*(xhatSample['AF',cc,i]+ahatSample['AF',cc,l]*(if xhatSample['AF',cc,i] < xmax then 1 else 0))
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*VAF[i]*max(0,xhatSample['AF',cc,i]-1) + (1-q[i])*VAF[i]*xhatSample['AF',cc,i]
		)
	);
subject to conAFgen {cc in GENCONS['AF']}:
	(1-alpha)*thetaAF >= sum{i in 1..I} (
		-VAF[i]*xhatGen['AF',cc,i]-beta*xhatGen['AF',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatGen['AF',cc,l]*r[l]*(if xhatGen['AF',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*VAF[i]*max(0,xhatGen['AF',cc,i]-1+ahatGen['AF',cc,l]*(if xhatGen['AF',cc,i] < xmax then 1 else 0))
			+(1-q[i])*VAF[i]*(xhatGen['AF',cc,i]+ahatGen['AF',cc,l]*(if xhatGen['AF',cc,i] < xmax then 1 else 0))
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*VAF[i]*max(0,xhatGen['AF',cc,i]-1) + (1-q[i])*VAF[i]*xhatGen['AF',cc,i]
		)
	);
subject to conAFred1 {A in 0..Amax}: (1-alpha)*thetaAF >= sum{i in 1..I} zAF[i,A];
subject to conAFred2 {i in 1..I, x in 0..xmax, A in 0..Amax}:
	zAF[i,A] >= -VAF[i]*x - beta*x 
	+p[i]*sum{l in 1..L}lambda[l]*floor((A mod 2^l)/(2^(l-1)))*r[l]*(if x<xmax then 1 else 0)
	+p[i]*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*VAF[i]*max(0,x-1+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0))
		+(1-q[i])*VAF[i]*(x+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0))
	)
	+(1-p[i])*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*VAF[i]*max(0,x-1)
		+(1-q[i])*VAF[i]*x
	);
subject to conAFadditional:
	thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax
	>= -beta*I*xmax/(1-alpha);

problem AFsample: ZAF, VAF, thetaAF, conAFsample, conAFadditional;
problem AFgen: ZAF, VAF, thetaAF, conAFgen, conAFadditional;
problem AFred: ZAF, VAF, thetaAF, zAF, conAFred1, conAFred2;
