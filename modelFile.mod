
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

set METHODS; #set of methods, maximal {'AFc','AF','SPL','Exact','acceptAll'}
set METHODSLP := METHODS diff {'Exact','acceptAll'}; #only ALP methods

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
param vSPLsim {i in 1..I, x in 0..xmax};
param VAFsim {i in 1..I};
param thetaAFsim;
param VAFcsim;
param thetaAFcsim;

#---------------------------------------------------------

var vSPL {i in 1..I, x in 0..xmax};
var zSPL {i in 1..I, A in 0..Amax};
var VAF {i in 1..I} <= 0;
var thetaAF;
var zAF {i in 1..I, A in 0..Amax};
var VAFc;
var thetaAFc;

minimize ZAFc: thetaAFc + 1/2*I*VAFc*xmax;
subject to conAFcred1 {A in 0..Amax}: (1-alpha)*thetaAFc >= sum{i in 1..I} zAF[i,A];
subject to conAFcred2 {i in 1..I, x in 0..xmax, A in 0..Amax}:
	zAF[i,A] >= -VAFc*x - beta*x 
	+p[i]*sum{l in 1..L}lambda[l]*floor((A mod 2^l)/(2^(l-1)))*r[l]*(if x<xmax then 1 else 0)
	+p[i]*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*VAFc*max(0,x-1+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0))
		+(1-q[i])*VAFc*(x+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0))
	)
	+(1-p[i])*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*VAFc*max(0,x-1)
		+(1-q[i])*VAFc*x
	);
subject to conAFcsample1 {cc in SAMCONS['AFc']}:
	(1-alpha)*thetaAFc >= sum{i in 1..I} (
		-VAFc*xhatSample['AFc',cc,i]-beta*xhatSample['AFc',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatSample['AFc',cc,l]*r[l]*(if xhatSample['AFc',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*VAFc*max(0,xhatSample['AFc',cc,i]-1+ahatSample['AFc',cc,l]*(if xhatSample['AFc',cc,i] < xmax then 1 else 0))
			+(1-q[i])*VAFc*(xhatSample['AFc',cc,i]+ahatSample['AFc',cc,l]*(if xhatSample['AFc',cc,i] < xmax then 1 else 0))
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*VAFc*max(0,xhatSample['AFc',cc,i]-1) + (1-q[i])*VAFc*xhatSample['AFc',cc,i]
		)
	);
subject to conAFcsample2: thetaAFc + 1/2*I*VAFc*xmax
	>= -beta*I*xmax/(1-alpha);
subject to conAFcgen1 {cc in GENCONS['AFc']}:
	(1-alpha)*thetaAFc >= sum{i in 1..I} (
		-VAFc*xhatGen['AFc',cc,i]-beta*xhatGen['AFc',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatGen['AFc',cc,l]*r[l]*(if xhatGen['AFc',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*VAFc*max(0,xhatGen['AFc',cc,i]-1+ahatGen['AFc',cc,l]*(if xhatGen['AFc',cc,i] < xmax then 1 else 0))
			+(1-q[i])*VAFc*(xhatGen['AFc',cc,i]+ahatGen['AFc',cc,l]*(if xhatGen['AFc',cc,i] < xmax then 1 else 0))
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*VAFc*max(0,xhatGen['AFc',cc,i]-1) + (1-q[i])*VAFc*xhatGen['AFc',cc,i]
		)
	);
subject to conAFcgen2:
	thetaAFc + 1/2*sum{i in 1..I}VAFc*xmax
	>= -beta*I*xmax/(1-alpha);

minimize ZAF: thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax;
subject to conAF1 {X in 0..Xmax, A in 0..Amax}:
	(1-alpha)*thetaAF >= -sum{i in 1..I}VAF[i]*floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
	-beta*sum{i in 1..I}floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
	+sum{l in 1..L, i in 1..I}lambda[l]*p[i]*(
		floor((A mod 2^l)/(2^(l-1)))*r[l]*(if floor((X mod (xmax+1)^i)/((xmax+1)^(i-1))) < xmax then 1 else 0)
		+alpha*sum{K in 0..Kmax}(
			prod{ii in 1..I}(q[ii]^(floor((K mod 2^ii)/(2^(ii-1))))*(1-q[ii])^(1-floor((K mod 2^ii)/(2^(ii-1)))))
		)*(
			sum{ii in 1..I}VAF[ii]*max(0,floor((X mod (xmax+1)^ii)/((xmax+1)^(ii-1)))
				-floor((K mod 2^ii)/(2^(ii-1)))
				+floor((A mod 2^l)/(2^(l-1)))*(if ii=i and floor((X mod (xmax+1)^ii)/((xmax+1)^(ii-1)))<xmax then 1 else 0))
		)
	);
subject to conAFsample1 {cc in SAMCONS['AF']}:
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
subject to conAFsample2: thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax 
	>= -beta*I*xmax/(1-alpha);
subject to conAFgen1 {cc in GENCONS['AF']}:
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
subject to conAFgen2:
	thetaAF + 1/2*sum{i in 1..I}VAF[i]*xmax
	>= -beta*I*xmax/(1-alpha);
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

minimize ZSPL: sum{i in 1..I, x in 0..xmax} 1/(xmax+1)*vSPL[i,x];
subject to conSPL1 {X in 0..Xmax, A in 0..Amax}:
	0 >= -sum{i in 1..I}vSPL[i,floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))]
	-beta*sum{i in 1..I}floor((X mod (xmax+1)^i)/((xmax+1)^(i-1)))
	+sum{l in 1..L, i in 1..I}lambda[l]*p[i]*(
		floor((A mod 2^l)/(2^(l-1)))*r[l]*(if floor((X mod (xmax+1)^i)/((xmax+1)^(i-1))) < xmax then 1 else 0)
		+alpha*sum{K in 0..Kmax}(
			prod{ii in 1..I}(q[ii]^(floor((K mod 2^ii)/(2^(ii-1))))*(1-q[ii])^(1-floor((K mod 2^ii)/(2^(ii-1)))))
		)*(
			sum{ii in 1..I}vSPL[ii,max(0,floor((X mod (xmax+1)^ii)/((xmax+1)^(ii-1)))
				-floor((K mod 2^ii)/(2^(ii-1)))
				+floor((A mod 2^l)/(2^(l-1)))*(if ii=i and floor((X mod (xmax+1)^ii)/((xmax+1)^(ii-1)))<xmax then 1 else 0))]
		)
	);
subject to conSPLsample1 {cc in SAMCONS['SPL']}:
	0 >= sum{i in 1..I} (
		-vSPL[i,xhatSample['SPL',cc,i]]-beta*xhatSample['SPL',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatSample['SPL',cc,l]*r[l]*(if xhatSample['SPL',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*vSPL[i,max(0,xhatSample['SPL',cc,i]-1+ahatSample['SPL',cc,l]*(if xhatSample['SPL',cc,i] < xmax then 1 else 0))]
			+(1-q[i])*vSPL[i,xhatSample['SPL',cc,i]+ahatSample['SPL',cc,l]*(if xhatSample['SPL',cc,i] < xmax then 1 else 0)]
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*vSPL[i,max(0,xhatSample['SPL',cc,i]-1)] + (1-q[i])*vSPL[i,xhatSample['SPL',cc,i]]
		)
	);
subject to conSPLsample2: sum{i in 1..I, x in 0..xmax} 1/(xmax+1)*vSPL[i,x]
	>= -beta*I*xmax/(1-alpha);
subject to conSPLgen1 {cc in GENCONS['SPL']}:
	0 >= sum{i in 1..I} (
		-vSPL[i,xhatGen['SPL',cc,i]]-beta*xhatGen['SPL',cc,i]
		+p[i]*sum{l in 1..L}lambda[l]*ahatGen['SPL',cc,l]*r[l]*(if xhatGen['SPL',cc,i] < xmax then 1 else 0)
		+alpha*p[i]*sum{l in 1..L}lambda[l]*(
			q[i]*vSPL[i,max(0,xhatGen['SPL',cc,i]-1+ahatGen['SPL',cc,l]*(if xhatGen['SPL',cc,i] < xmax then 1 else 0))]
			+(1-q[i])*vSPL[i,xhatGen['SPL',cc,i]+ahatGen['SPL',cc,l]*(if xhatGen['SPL',cc,i] < xmax then 1 else 0)]
		)
		+alpha*(1-p[i])*sum{l in 1..L}lambda[l]*(
			q[i]*vSPL[i,max(0,xhatGen['SPL',cc,i]-1)] + (1-q[i])*vSPL[i,xhatGen['SPL',cc,i]]
		)
	);
subject to conSPLgen2:
	sum{i in 1..I, x in 0..xmax} 1/(xmax+1)*vSPL[i,x]
	>= -beta*I*xmax/(1-alpha);
subject to conSPLred1 {A in 0..Amax}: 0 >= sum{i in 1..I} zSPL[i,A];
subject to conSPLred2 {i in 1..I, x in 0..xmax, A in 0..Amax}:
	zSPL[i,A] >= -vSPL[i,x] - beta*x 
	+p[i]*sum{l in 1..L}lambda[l]*floor((A mod 2^l)/(2^(l-1)))*r[l]*(if x<xmax then 1 else 0)
	+p[i]*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*vSPL[i,max(0,x-1+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0))]
		+(1-q[i])*vSPL[i,x+floor((A mod 2^l)/(2^(l-1)))*(if x<xmax then 1 else 0)]
	)
	+(1-p[i])*alpha*sum{l in 1..L}lambda[l]*(
		q[i]*vSPL[i,max(0,x-1)]
		+(1-q[i])*vSPL[i,x]
	);

problem SPL: ZSPL, vSPL, conSPL1;
problem SPLsample: ZSPL, vSPL, conSPLsample1, conSPLsample2;
problem SPLgen: ZSPL, vSPL, conSPLgen1, conSPLgen2;
problem SPLred: ZSPL, vSPL, zSPL, conSPLred1, conSPLred2;
problem AF: ZAF, VAF, thetaAF, conAF1;
problem AFsample: ZAF, VAF, thetaAF, conAFsample1, conAFsample2;
problem AFgen: ZAF, VAF, thetaAF, conAFgen1, conAFgen2;
problem AFred: ZAF, VAF, thetaAF, zAF, conAFred1, conAFred2;
problem AFcsample: ZAFc, VAFc, thetaAFc, conAFcsample1, conAFcsample2;
problem AFcgen: ZAFc, VAFc, thetaAFc, conAFcgen1, conAFcgen2;
problem AFcred: ZAFc, VAFc, thetaAFc, zAF, conAFcred1, conAFcred2;