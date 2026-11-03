function run_reference(infile, outfile, utildir)
% Run the original vbPET MATLAB code on the problem stored in INFILE and save
% model predictions and VB estimates to OUTFILE (used by the Python tests).
addpath(utildir);
S = load(infile);

info.tp = S.tp(:);
info.Cp = S.Cp(:);
info.Cb = S.Cb(:);
info.t  = S.t(:);
info.weights = S.weights(:);
info.phi.s0 = S.s0;
info.phi.c0 = S.c0;
info.ARD = S.ard(:)';
info.checkARD = @(a, m0, t) a;
info.dbf = 0;
info.dfig = 0;
info.derivMethod = 'numeric';
info.computeModel = str2func(['computeModel_' S.model]);
info.derivedPar = str2func(['Derived_Par_' S.model]);
info.parName = 'derived';

% model predictions for a set of parameter vectors
P = S.theta_pred;
pred = zeros(numel(info.t), size(P, 2));
for i = 1:size(P, 2)
    pred(:, i) = info.computeModel(P(:, i), info);
end

m0 = S.m0(:);
L0 = diag(1 ./ ((S.lambda * m0).^2));
tic;
R = vbEstimation(m0, L0, size(S.Z, 2), S.Z, info);
elapsed = toc;

out.pred = pred;
out.par = R.par;
out.COV = R.COV;
out.FreeEnergy = R.FreeEnergy;
out.s = R.s;
out.c = R.c;
out.num_iter = R.num_iter;
out.fit = R.fit;
out.elapsed = elapsed;
save(outfile, '-struct', 'out');
end
