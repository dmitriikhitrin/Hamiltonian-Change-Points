% Quantum state certification procedure from the following paper:
%   Few Single-Qubit Measurements Suffice to Certify Any Quantum State
%   Meghal Gupta, William He, Ryan O'Donnell
% The hypothesis state is accessed both directly and through oracles.
% These two methods are asserted to be equivalent throughout the algorithm.

function acc = certify(n, hyparg, lab)
	assert(all(size(lab) == size(hyparg)));
	assert(all(size(lab) == [2^n, 1]));
	assert(abs(1-norm(hyparg)) < 1e-12);
	assert(abs(1-norm(lab)) < 1e-12);

	global P hyp
	hyp = hyparg;
	P = cell(n,1);
	for i = 1:n
		P{i} = eye(2);
	end

	acc = 0;
	for k = 1:n
		acc = acc + left(1, k, n, hyp, lab) / n;
	end
end

function acc = left(i, k, n, hyparg, lab)
	global P
	if i == k
		acc = right(i+1, k, n, hyparg, lab);
	else
		% Measure qubits to the left of kth (1st)
		acc = 0;
		for c = [[1;0],[0;1]]
			P{i} = c * c';

			hypc = kron(c, eye(2^(n-i)))' * hyparg;
			phc = sum(conj(hypc) .* hypc);
			hypc = hypc ./ sqrt(phc);

			labc = kron(c, eye(2^(n-i)))' * lab;
			plc = sum(conj(labc) .* labc);
			labc = labc ./ sqrt(plc);

			if plc > 0 && phc > 0
				acc = acc + plc * left(i+1, k, n, hypc, labc);
			end
		end
		P{i} = eye(2);
	end
end

function acc = right(i, k, n, hyparg, lab)
	global P hyp
	bas = [[1;1]/sqrt(2),[1;1i]/sqrt(2),[1;0]];
	if i > n
		% Base case: measure kth qubit
		crd = zeros(1,3);
		cf = oracle(hyp, P);
		for xi = 1:3
			x = bas(:,xi);
			P{k} = x*x';
			crd(xi) = 2*oracle(hyp, P)/cf-1;
		end
		P{k} = eye(2);
		fin = blo2vec(crd);
		acc = abs(fin' * lab)^2;

		accp = abs(hyparg' * lab)^2;
		assert(norm(acc - accp) < 1e-12);
	else
		% Measure qubits to the right of kth (k+1th)
		hyp0 = kron([1;0], eye(2^(n-i+1)))' * hyparg;
		ph0 = sqrt(sum(conj(hyp0) .* hyp0));
		if ph0 == 0
			hyp0 = RandomStateVector(2^(n-i+1));
		else
			hyp0 = hyp0 ./ ph0;
		end
		rho0 = PartialTrace(hyp0*hyp0',2,[2,2^(n-i)]);
		blo0 = mat2blo(rho0);

		hyp1 = kron([0;1], eye(2^(n-i+1)))' * hyparg;
		ph1 = sqrt(sum(conj(hyp1) .* hyp1));
		if ph1 == 0
				hyp1 = RandomStateVector(2^(n-i+1));
		else
			hyp1 = hyp1 ./ ph1;
		end
		rho1 = PartialTrace(hyp1*hyp1',2,[2,2^(n-i)]);
		blo1 = mat2blo(rho1);

		P{k} = [1;0] * [1;0]';
		crd0 = zeros(1,3);
		cf = oracle(hyp, P);
		if cf > 0
			for xi = 1:3
				x = bas(:,xi);
				P{i} = x*x';
				crd0(xi) = 2*oracle(hyp, P)/cf-1;
			end
			P{i} = eye(2);
			assert(norm(crd0 - blo0) < 1e-12);
		end

		P{k} = [0;1] * [0;1]';
		crd1 = zeros(1,3);
		cf = oracle(hyp, P);
		if cf > 0
			for xi = 1:3
				x = bas(:,xi);
				P{i} = x*x';
				crd1(xi) = 2*oracle(hyp, P)/cf-1;
			end
			P{i} = eye(2);
			assert(norm(crd1 - blo1) < 1e-12);
		end

		P{k} = eye(2);

		nul = null([crd0; crd1]);
		rnd = randn(size(nul,2),1);
		temp = (nul * rnd ./ norm(rnd))';
		acc = 0;
		for b = [blo2vec(temp),blo2vec(-temp)]
			P{i} = b * b';

			hypb = kron(kron(eye(2), b), eye(2^(n-i)))' * hyparg;
			phb = sum(conj(hypb) .* hypb);
			assert(abs(phb - 1/2) < 1e-12);
			hypb = hypb ./ sqrt(phb);

			labb = kron(kron(eye(2), b), eye(2^(n-i)))' * lab;
			plb = sum(conj(labb) .* labb);
			labb = labb ./ sqrt(plb);

			if plb > 0
				acc = acc + plb * right(i+1, k, n, hypb, labb);
			end
		end
		P{i} = eye(2);
	end
end

function blo = mat2blo(rho)
	blo = arrayfun(@(i) trace(rho * Pauli(i)), 1:3);
end

function v = blo2vec(blo)
	assert(1 - norm(blo) < 1e-12);
	rho = (eye(2) + blo(1)*Pauli(1) + blo(2)*Pauli(2) + blo(3)*Pauli(3))/2;
	[v,~] = eigs(rho,1);
end

function inp = oracle(phi, P)
	% Validate phi
	assert(size(phi,2) == 1);
	N = size(phi,1);
	assert(~bitand(N, N-1));
	n = log2(N);

	% Validate P
	assert(all(size(P) == [n 1]));
	for i = 1:n
		Pi = P{i};
		assert(all(size(Pi) == [2 2]));
		assert(all(ismembertol(eig(Pi), [0 1], 1e-12)));
	end

	% Compute inner product
	inp = phi' * krn(P{:}) * phi;
end
